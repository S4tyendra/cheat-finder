from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Create directories if they don't exist
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def create_similarity_heatmap(similarity_data, file_names):
    """Create an enhanced heatmap of document similarities"""
    df = pd.DataFrame(similarity_data, index=file_names, columns=file_names)
    
    # Create heatmap with improved styling
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
    
    # Add edges (connections)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
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
    
    # Split into sentences
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
    <html>
    <head>
        <title>Document Similarity Analyzer</title>
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #3498db;
                --accent-color: #e74c3c;
                --background-color: #f8f9fa;
                --card-background: #ffffff;
            }
            
            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                margin: 0;
                padding: 0;
                background-color: var(--background-color);
                color: var(--primary-color);
                min-height: 100vh;
            }
            
            .header {
                background-color: var(--primary-color);
                color: white;
                padding: 2rem;
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .upload-card {
                background: var(--card-background);
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 2rem;
                margin-bottom: 2rem;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            
            .file-input-container {
                position: relative;
                margin: 1rem 0;
                padding: 2rem;
                border: 2px dashed var(--secondary-color);
                border-radius: 8px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .file-input-container:hover {
                border-color: var(--primary-color);
                background-color: rgba(52, 152, 219, 0.05);
            }
            
            .file-input {
                font-size: 1rem;
                width: 100%;
            }
            
            .button {
                background-color: var(--secondary-color);
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 4px;
                font-size: 1.1rem;
                cursor: pointer;
                transition: background-color 0.3s ease;
                width: 100%;
            }
            
            .button:hover {
                background-color: #2980b9;
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-top: 3rem;
            }
            
            .feature-card {
                background: var(--card-background);
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .feature-card h3 {
                color: var(--secondary-color);
                margin-top: 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Document Similarity Analyzer</h1>
        </div>
        <div class="container">
            <div class="upload-card">
                <form action="/analyze/" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label>Upload Your Documents</label>
                        <div class="file-input-container">
                            <input type="file" name="files" multiple accept=".txt,.doc,.docx,.pdf" required class="file-input">
                            <p>Drag and drop files here or click to select</p>
                            <small>Maximum 150 files supported</small>
                        </div>
                    </div>
                    <button type="submit" class="button">Analyze Documents</button>
                </form>
            </div>
            
            <div class="features">
                <div class="feature-card">
                    <h3>Advanced Analysis</h3>
                    <p>Sophisticated algorithms to detect document similarities and potential content matches.</p>
                </div>
                <div class="feature-card">
                    <h3>Visual Results</h3>
                    <p>Interactive heatmaps and network graphs to visualize document relationships.</p>
                </div>
                <div class="feature-card">
                    <h3>Detailed Metrics</h3>
                    <p>Comprehensive similarity scores and content analysis for each document.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze/")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """Analyze uploaded documents for similarity"""
    if len(files) > 150:
        raise HTTPException(status_code=400, detail="Maximum 150 files allowed")
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        for file in files:
            suffix = Path(file.filename or "").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(Path(temp_file.name))
        
        # Create similarity matrix
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
                
                # If similarity is high, find similar content
                if similarity_score > 0.7:
                    similar_pairs = find_similar_content(temp_files[i], temp_files[j])
                    if similar_pairs:
                        similar_content.append({
                            'file1': files[i].filename,
                            'file2': files[j].filename,
                            'similar_pairs': similar_pairs
                        })
        
        # Create visualizations
        heatmap_html = create_similarity_heatmap(similarity_matrix, [file.filename for file in files])
        network_html = create_similarity_network(similarity_matrix, [file.filename for file in files])
        
        # Calculate document metrics
        metrics = calculate_document_metrics(similarity_matrix, [file.filename for file in files])
        
        # Create result HTML
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
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)