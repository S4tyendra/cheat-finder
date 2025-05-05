# Document Similarity Analyzer

## Problem Statement
In academic and professional environments, detecting document similarities and potential plagiarism is crucial for maintaining integrity. Manual comparison of multiple documents is time-consuming and prone to errors. This project addresses this challenge by providing an automated solution for:
- Analyzing similarities between multiple documents simultaneously
- Visualizing document relationships
- Detecting potentially copied content
- Providing quantitative metrics for document uniqueness

## Features
### Core Analysis
- Multiple similarity detection methods:
  - Cosine similarity with TF-IDF vectorization
  - Cosine similarity with Count vectorization
  - Jaccard similarity (token-based)
  - Jaccard similarity (n-gram based)
- Advanced text preprocessing:
  - Tokenization
  - Stop word removal
  - Case normalization
  - Support for stemming/lemmatization

### Visualization and Reporting
- Interactive heatmap of document similarities
- Network graph showing document relationships
- Detailed metrics for each document:
  - Average similarity score
  - Maximum similarity score
  - Unique content score
- Highlighted potentially copied content sections

### User Interface
- Modern, responsive web interface
- Drag-and-drop file upload
- Support for multiple file formats:
  - Plain text (.txt)
  - Microsoft Word (.doc, .docx)
  - PDF documents (.pdf)
- Real-time analysis results
- Maximum support for 150 files simultaneously

## Technical Architecture
### Backend
- FastAPI framework for high-performance API
- NLTK for natural language processing
- scikit-learn for text vectorization and similarity calculations
- Plotly for interactive visualizations

### Frontend
- Pure HTML/CSS for lightweight interface
- Interactive visualizations using Plotly
- Responsive design for various screen sizes

## Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/S4tyendra/cheat-finder.git
cd cheat-finder
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

5. Access the web interface:
   - Open your browser and navigate to `http://localhost:8000`

## Usage
1. Access the web interface through your browser
2. Upload documents using the file upload form
   - Drag and drop files or click to select
   - Select multiple files (up to 150)
3. Click "Analyze Documents" to start the analysis
4. View the results:
   - Similarity heatmap
   - Network visualization
   - Document metrics table
   - Potential copied content sections

## Implementation Details
### Similarity Metrics
1. **TF-IDF Cosine Similarity**
   - Considers word importance in documents
   - Reduces impact of common words
   - Range: 0 (different) to 1 (identical)

2. **Count Vectorizer Cosine Similarity**
   - Based on word frequency
   - Simpler but effective approach
   - Range: 0 (different) to 1 (identical)

3. **Jaccard Similarity**
   - Token-based comparison
   - N-gram based comparison
   - Range: 0 (no overlap) to 1 (identical)

### Text Preprocessing
- Tokenization using NLTK
- Stop word removal
- Case normalization
- Optional stemming/lemmatization

## Future Enhancements
- Support for more file formats
- Advanced language detection
- Customizable similarity thresholds
- Batch processing capabilities
- Export results in various formats
- API documentation using Swagger UI

## Contributors
- [S4tyendra](https://github.com/S4tyendra)
