from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tempfile
from typing import List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from analyzer import analyze_similarity, load_document

app = FastAPI(title="Document Similarity Analyzer")

# Create a templates directory if it doesn't exist
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def create_similarity_plot(similarity_data):
    """Create a heatmap of document similarities"""
    df = pd.DataFrame(similarity_data)
    fig = px.imshow(df,
                    labels=dict(x="Document 2", y="Document 1", color="Similarity"),
                    title="Document Similarity Heatmap")
    return fig.to_html(full_html=False)

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
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin-bottom: 20px; }
            .button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .results { margin-top: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Document Similarity Analyzer</h1>
            <form action="/analyze/" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Upload Documents (Max 150 files):</label><br>
                    <input type="file" name="files" multiple accept=".txt,.doc,.docx,.pdf" required>
                </div>
                <button type="submit" class="button">Analyze Documents</button>
            </form>
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
        
        # Create visualization
        plot_html = create_similarity_plot(similarity_matrix)
        
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Results</h1>
                <div class="similarity-plot">
                    <h2>Similarity Heatmap</h2>
                    {plot_html}
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