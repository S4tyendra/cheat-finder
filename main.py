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
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# Create static folder if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)



templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Document Similarity Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.post("/analyze/")
async def analyze_documents(request: Request, files: List[UploadFile] = File(...)):
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
        
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "similarity_matrix": similarity_matrix,
                "heatmap_html": heatmap_html,
                "network_html": network_html,
                "metrics": metrics,
                "similar_content": similar_content
            }
        )
        
    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
