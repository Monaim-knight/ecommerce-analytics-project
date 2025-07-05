import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import re

def markdown_to_pdf(markdown_file, pdf_file):
    """Convert markdown file to PDF using reportlab"""
    
    # Read the markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.darkblue
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leftIndent=20,
        backColor=colors.lightgrey
    )
    
    # Process HTML content
    lines = html_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
            
        # Handle headers
        if line.startswith('<h1>'):
            text = re.sub(r'<h1>(.*?)</h1>', r'\1', line)
            story.append(Paragraph(text, title_style))
            story.append(Spacer(1, 12))
            
        elif line.startswith('<h2>'):
            text = re.sub(r'<h2>(.*?)</h2>', r'\1', line)
            story.append(Paragraph(text, heading_style))
            story.append(Spacer(1, 8))
            
        elif line.startswith('<h3>'):
            text = re.sub(r'<h3>(.*?)</h3>', r'\1', line)
            story.append(Paragraph(text, subheading_style))
            story.append(Spacer(1, 6))
            
        # Handle code blocks
        elif line.startswith('<pre><code>'):
            text = re.sub(r'<pre><code>(.*?)</code></pre>', r'\1', line, flags=re.DOTALL)
            story.append(Paragraph(text, code_style))
            story.append(Spacer(1, 6))
            
        # Handle regular paragraphs
        elif line.startswith('<p>'):
            text = re.sub(r'<p>(.*?)</p>', r'\1', line)
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 6))
            
        # Handle lists
        elif line.startswith('<li>'):
            text = re.sub(r'<li>(.*?)</li>', r'• \1', line)
            story.append(Paragraph(text, styles['Normal']))
            
        # Handle other content
        else:
            if line:
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    print(f"PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    markdown_to_pdf("INTERVIEW_PREPARATION_GUIDE.md", "E-commerce_Analytics_Interview_Guide.pdf") 