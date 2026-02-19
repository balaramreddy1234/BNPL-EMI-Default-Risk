"""
PDF extraction utility for extracting tabular data from PDF files
"""
import PyPDF2
import re
import pandas as pd
from io import StringIO

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file
    Returns: raw text content
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def parse_csv_from_pdf(pdf_text):
    """
    Parse CSV-like content from PDF text
    Assumes a structured format with column headers and rows
    Returns: pandas DataFrame
    """
    try:
        # Try to extract CSV-like content
        lines = pdf_text.strip().split('\n')
        
        if not lines:
            raise ValueError("No content found in PDF")
        
        # Create CSV-like content
        csv_content = '\n'.join(lines)
        
        # Try to parse as CSV
        df = pd.read_csv(StringIO(csv_content), sep=None, engine='python')
        
        return df
    except Exception as e:
        raise Exception(f"Error parsing PDF as tabular data: {str(e)}")

def extract_data_from_pdf(pdf_path):
    """
    Extract and parse tabular data from PDF
    Returns: pandas DataFrame with EMI data
    """
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Parse as CSV
        df = parse_csv_from_pdf(text)
        
        return df
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")
