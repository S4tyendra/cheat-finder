from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import os
import tempfile
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from analyzer import analyze_similarity, load_document

app = FastAPI(title="Document Similarity Analyzer")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



def create_similarity_heatmap(similarity_data, file_names):
    """Create an enhanced heatmap of document similarities"""
    df = pd.DataFrame(similarity_data, index=file_names, columns=file_names)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdYlBu_r',
        zmin=0,
        zmax=1,
        hoverongaps=False,
        hovertemplate='Document 1: %{y}<br>Document 2: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Document Similarity Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        margin=dict(t=100, l=100, r=50, b=50),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_html(full_html=False, config={'displayModeBar': True})

def create_similarity_network(similarity_data, file_names, threshold=0.3):
    """Create a network graph of document similarities"""
    n_files = len(file_names)
    edges = []
    edge_weights = []
    
    for i in range(n_files):
        for j in range(i+1, n_files):
            if similarity_data[i][j] > threshold:
                edges.append((i, j))
                edge_weights.append(similarity_data[i][j])
    
    edge_x = []
    edge_y = []
    for edge in edges:
        x0 = np.cos(2*np.pi*edge[0]/n_files)
        y0 = np.sin(2*np.pi*edge[0]/n_files)
        x1 = np.cos(2*np.pi*edge[1]/n_files)
        y1 = np.sin(2*np.pi*edge[1]/n_files)
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [np.cos(2*np.pi*i/n_files) for i in range(n_files)]
    node_y = [np.sin(2*np.pi*i/n_files) for i in range(n_files)]
    
    fig = go.Figure()
    
    # edges (connections)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='#1f77b4',
            line=dict(width=2, color='#fff')
        ),
        text=file_names,
        textposition="top center",
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title={
            'text': 'Document Similarity Network',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(t=100, l=50, r=50, b=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_html(full_html=False, config={'displayModeBar': True})

def calculate_document_metrics(similarity_matrix: List[List[float]], file_names: List[str]) -> List[Dict]:
    """Calculate various metrics for each document"""
    n_files = len(similarity_matrix)
    metrics = []
    
    for i in range(n_files):
        similarities = similarity_matrix[i]
        metrics.append({
            'name': file_names[i],
            'avg_similarity': sum(similarities) / (n_files - 1),
            'max_similarity': max(s for j, s in enumerate(similarities) if i != j),
            'unique_score': 1 - (sum(similarities) / (n_files - 1))
        })
    
    return metrics

def find_similar_content(doc1_path: Path, doc2_path: Path, threshold: float = 0.8):
    """Find potentially copied content between documents"""
    doc1_content = load_document(doc1_path)
    doc2_content = load_document(doc2_path)
    
    if doc1_content is None or doc2_content is None:
        return []
    
    # Split as sentences
    doc1_sentences = [s.strip() for s in doc1_content.split('.') if s.strip()]
    doc2_sentences = [s.strip() for s in doc2_content.split('.') if s.strip()]
    
    similar_pairs = []
    for i, sent1 in enumerate(doc1_sentences):
        for j, sent2 in enumerate(doc2_sentences):
            similarity = analyze_similarity(sent1, sent2)
            if similarity.get('Cosine (TF-IDF)', 0) > threshold:
                similar_pairs.append({
                    'doc1_sentence': sent1,
                    'doc2_sentence': sent2,
                    'similarity': similarity['Cosine (TF-IDF)']
                })
    
    return similar_pairs

@app.get("/", response_class=HTMLResponse)
async def home():
    """Render the home page with file upload form"""
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Similarity Analyzer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Reset and Base Styles */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

:root {
    --background-color: #121212; /* Dark background */
    --card-background: rgba(40, 40, 40, 0.6); /* Semi-transparent dark card */
    --backdrop-blur: 10px; /* Blur effect for glassmorphism */
    --border-color: rgba(255, 255, 255, 0.1);
    --text-color: #e0e0e0; /* Light text */
    --text-secondary: #a0a0a0; /* Dimmed text */
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Purple-blue gradient */
    --accent-color: #667eea; /* Primary color from gradient */
    --hover-brightness: 1.15;
    --border-radius: 12px; /* Softer corners */
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
    line-height: 1.6;
    background-image: url('background.png'); /* Optional: Subtle background pattern */
    background-size: cover;
    background-attachment: fixed;
    overflow-x: hidden; /* Prevent horizontal scroll */
    min-height: 100vh;
}

/* Header */
.header {
    padding: 3rem 1rem 2rem;
    text-align: center;
    background: rgba(0, 0, 0, 0.2); /* Slight overlay */
    margin-bottom: 3rem;
    position: relative; /* For potential pseudo-elements */
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-fill-color: transparent; /* Apply gradient to text */
    display: inline-block; /* Needed for text gradient */
}

.header .subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 300;
}

/* Container */
.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 2rem 4rem; /* Add bottom padding */
}

/* Glassmorphism Card Style */
.glassmorphism {
    background: var(--card-background);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
    backdrop-filter: blur(var(--backdrop-blur));
    -webkit-backdrop-filter: blur(var(--backdrop-blur)); /* Safari support */
    box-shadow: var(--shadow);
    padding: 2rem;
    margin-bottom: 2.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glassmorphism:hover {
    /* Optional: Slight lift effect on hover */
    /* transform: translateY(-5px); */
    /* box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4); */
}

.upload-card {
    text-align: center;
}

.form-group {
    margin-bottom: 1.5rem;
    text-align: left;
}

.form-group label {
    display: block;
    margin-bottom: 0.75rem;
    font-weight: 500;
    font-size: 1.1rem;
    color: var(--text-color);
}

.file-input-container {
    position: relative;
    padding: 2.5rem 1.5rem;
    border: 2px dashed var(--border-color);
    border-radius: var(--border-radius);
    text-align: center;
    cursor: pointer;
    transition: border-color 0.3s ease, background-color 0.3s ease;
}

.file-input-container:hover {
    border-color: var(--accent-color);
    background-color: rgba(102, 126, 234, 0.1);
}

.upload-icon {
    width: 48px;
    height: 48px;
    stroke: var(--accent-color);
    margin-bottom: 1rem;
}

.file-input {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
}

.drag-drop-text {
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    color: var(--text-color);
}

.drag-drop-text strong {
    color: var(--accent-color);
    font-weight: 600;
}

.file-info {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

#file-list {
    margin-top: 1rem;
    text-align: left;
    font-size: 0.9rem;
    color: var(--text-secondary);
}
#file-list div {
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.button {
    display: inline-flex; 
    align-items: center;
    justify-content: center;
    gap: 0.75rem; 
    color: white;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: var(--border-radius);
    font-size: 1.1rem;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.2s ease, filter 0.3s ease;
    width: 100%;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.gradient-button {
    background: var(--primary-gradient);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.button:hover {
    filter: brightness(var(--hover-brightness));
    transform: translateY(-2px); 
}

.button:active {
    transform: translateY(0);
    filter: brightness(1);
}

.button-icon {
    width: 20px;
    height: 20px;
    stroke-width: 2;
}

.features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-top: 3rem;
}

.feature-card {
    text-align: center; 
}

.feature-icon {
    width: 40px;
    height: 40px;
    stroke: var(--accent-color);
    margin-bottom: 1rem;
}

.feature-card h3 {
    color: var(--text-color);
    margin-bottom: 0.75rem;
    font-size: 1.3rem;
    font-weight: 600;
}

.feature-card p {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

@media (max-width: 768px) {
    .header h1 {
        font-size: 2.5rem;
    }
    .container {
        padding: 0 1rem 3rem;
    }
    .glassmorphism {
        padding: 1.5rem;
    }
    .features {
        grid-template-columns: 1fr; 
    }
    .button {
        font-size: 1rem;
        padding: 0.8rem 1.5rem;
    }
    .file-input-container {
        padding: 2rem 1rem;
    }
}



    </style>
</head>
<body>
    <div class="header">
        <h1>Docu<span class="highlight">Sim</span> Analyzer</h1>
        <p class="subtitle">Modern Document Similarity Analysis</p>
    </div>
    <div class="container">
        <div class="upload-card glassmorphism">
            <form action="/analyze/" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file-upload">Upload Your Documents</label>
                    <div class="file-input-container">
                         <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="upload-icon">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
                         </svg>
                        <input type="file" id="file-upload" name="files" multiple accept=".txt,.doc,.docx,.pdf" required class="file-input">
                        <p class="drag-drop-text">Drag & drop files here or <strong>click to select</strong></p>
                        <small class="file-info">Supports: .txt, .doc, .docx, .pdf (Max 150 files)</small>
                        <div id="file-list"></div>
                    </div>
                </div>
                <button type="submit" class="button gradient-button">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="button-icon">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 004.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L4.2 15.3m15.6 0c1.636 0 2.962 1.326 2.962 2.962v.408c0 .836-.335 1.596-.876 2.14l-.992.992a.999.999 0 01-1.413 0l-1.172-1.172a.999.999 0 00-1.414 0l-.992.992a.749.749 0 01-1.06 0l-1.172-1.172a.749.749 0 00-1.06 0L12 19.468l-.992.992a.749.749 0 01-1.06 0l-1.172-1.172a.749.749 0 00-1.06 0l-.992.992a.999.999 0 01-1.414 0l-1.172-1.172a.999.999 0 00-1.414 0l-.992.992a.749.749 0 01-1.06 0l-1.172-1.172a.999.999 0 00-1.414 0l-1.57-.393m15.6 0c-3.436 0-6.23.693-6.23 1.562 0 .868 2.794 1.562 6.23 1.562 3.436 0 6.23-.694 6.23-1.562 0-.87-2.794-1.562-6.23-1.562z" />
                    </svg>
                    Analyze Documents
                </button>
            </form>
        </div>

        <div class="features">
            <div class="feature-card glassmorphism">
                 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="feature-icon">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                 </svg>
                <h3>Advanced Analysis</h3>
                <p>Sophisticated algorithms detect subtle similarities and potential content matches.</p>
            </div>
            <div class="feature-card glassmorphism">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="feature-icon">
                   <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                   <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <h3>Visual Results</h3>
                <p>Interactive heatmaps and network graphs visualize document relationships clearly.</p>
            </div>
            <div class="feature-card glassmorphism">
                 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="feature-icon">
                   <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
                 </svg>
                <h3>Detailed Metrics</h3>
                <p>Comprehensive similarity scores and detailed content analysis for each document pair.</p>
            </div>
        </div>
    </div>
    <script>
        const fileInput = document.getElementById('file-upload');
const fileListDisplay = document.getElementById('file-list');
const fileInputContainer = document.querySelector('.file-input-container');
const dragDropText = document.querySelector('.drag-drop-text');

function updateFileList() {
    fileListDisplay.innerHTML = ''; 
    if (fileInput.files.length > 0) {
        dragDropText.style.display = 'none'; // Hide initial text
        const list = document.createElement('div');
        list.innerHTML = `<strong>Selected Files (${fileInput.files.length}):</strong>`;
        for (let i = 0; i < fileInput.files.length; i++) {
            const fileItem = document.createElement('div');
            fileItem.textContent = `- ${fileInput.files[i].name}`;
            list.appendChild(fileItem);
        }
        fileListDisplay.appendChild(list);
    } else {
        dragDropText.style.display = 'block'; 
    }
}

fileInput.addEventListener('change', updateFileList);


['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  fileInputContainer.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false); // Prevent drop outside area
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
  fileInputContainer.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  fileInputContainer.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
  fileInputContainer.style.borderColor = 'var(--accent-color)';
  fileInputContainer.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
}

function unhighlight(e) {
  fileInputContainer.style.borderColor = 'var(--border-color)';
  fileInputContainer.style.backgroundColor = 'transparent';
}

fileInputContainer.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  fileInput.files = files;

  updateFileList();
}
updateFileList();


    </script>
</body>
</html>


    """
    return HTMLResponse(content=html_content)

@app.post("/analyze/")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """Analyze uploaded documents for similarity"""
    if len(files) > 150:
        raise HTTPException(status_code=400, detail="Maximum 150 files allowed")
    
    temp_files = []
    try:
        for file in files:
            suffix = Path(file.filename or "").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(Path(temp_file.name))
        
        # similarity matrix
        n_files = len(temp_files)
        similarity_matrix = [[0.0] * n_files for _ in range(n_files)]
        similar_content = []
        
        # Compare each pair of documents
        for i in range(n_files):
            for j in range(i + 1, n_files):
                similarity = analyze_similarity(
                    load_document(temp_files[i]),
                    load_document(temp_files[j])
                )
                similarity_score = similarity.get('Cosine (TF-IDF)', 0)
                similarity_matrix[i][j] = similarity_score
                similarity_matrix[j][i] = similarity_score
                
               
                if similarity_score > 0.7:
                    similar_pairs = find_similar_content(temp_files[i], temp_files[j])
                    if similar_pairs:
                        similar_content.append({
                            'file1': files[i].filename,
                            'file2': files[j].filename,
                            'similar_pairs': similar_pairs
                        })
        
        heatmap_html = create_similarity_heatmap(similarity_matrix, [file.filename for file in files])
        network_html = create_similarity_network(similarity_matrix, [file.filename for file in files])
        
        metrics = calculate_document_metrics(similarity_matrix, [file.filename for file in files]) # type: ignore
        
        # result HTML
        result_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .similarity-plot {{ margin: 20px 0; }}
                .similar-content {{ margin: 20px 0; }}
                .similar-pair {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Results</h1>
                <div class="similarity-plot">
                    <h2>Similarity Heatmap</h2>
                    {heatmap_html}
                </div>
                <div class="similarity-plot">
                    <h2>Similarity Network</h2>
                    {network_html}
                </div>
                <div class="metrics">
                    <h2>Document Metrics</h2>
                    <table class="metrics-table">
                        <tr>
                            <th>Document</th>
                            <th>Average Similarity</th>
                            <th>Max Similarity</th>
                            <th>Unique Score</th>
                        </tr>
                        {''.join(f"""
                        <tr>
                            <td>{metric['name']}</td>
                            <td>{metric['avg_similarity']:.2f}</td>
                            <td>{metric['max_similarity']:.2f}</td>
                            <td>{metric['unique_score']:.2f}</td>
                        </tr>
                        """ for metric in metrics)}
                    </table>
                </div>
                <div class="similar-content">
                    <h2>Potentially Copied Content</h2>
                    {''.join(f'''
                    <div class="similar-pair">
                        <h3>Similar content between {pair['file1']} and {pair['file2']}</h3>
                        {''.join(f"""
                        <p>Similarity: {match['similarity']:.2f}</p>
                        <p>Document 1: {match['doc1_sentence']}</p>
                        <p>Document 2: {match['doc2_sentence']}</p>
                        <hr>
                        """ for match in pair['similar_pairs'])}
                    </div>
                    ''' for pair in similar_content)}
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=result_html)
        
    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
