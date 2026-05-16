"""
create_technical_pdf.py

This script converts the Technical Analysis Document into a PDF file.

The final PDF is required by the project guideline.

Input:
    docs/technical_analysis.md

Output:
    docs/technical_analysis.pdf

The script uses ReportLab, a Python library for PDF generation.
"""

# Path is used to handle file paths in a clean way
from pathlib import Path

# ReportLab is used to generate the PDF file
from reportlab.lib.pagesizes import A4

# Import standard PDF styling tools
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Import colors for table and heading styling
from reportlab.lib import colors

# Import units for margins
from reportlab.lib.units import cm

# Import document layout classes
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle
)

# Import text alignment constants
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def clean_markdown_line(line):
    """
    Cleans a markdown line and prepares it for PDF rendering.

    Parameters:
        line: one line from the markdown file

    Returns:
        cleaned line
    """

    # Remove markdown bold markers
    line = line.replace("**", "")

    # Replace markdown inline code markers
    line = line.replace("`", "")

    # Replace problematic arrows with ASCII-compatible arrows
    line = line.replace("→", "->")

    return line


def create_styles():
    """
    Creates custom paragraph styles for the PDF.

    Returns:
        styles: dictionary of ReportLab styles
    """

    # Get default ReportLab styles
    base_styles = getSampleStyleSheet()

    # Main title style
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=base_styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=16
    )

    # Section heading style
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=base_styles["Heading1"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1f3a5f"),
        spaceBefore=12,
        spaceAfter=8
    )

    # Subsection heading style
    subheading_style = ParagraphStyle(
        "CustomSubHeading",
        parent=base_styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2f5597"),
        spaceBefore=10,
        spaceAfter=6
    )

    # Normal paragraph style
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=base_styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=5
    )

    # Bullet point style
    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=base_styles["Normal"],
        fontSize=9,
        leading=12,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=3
    )

    # Code block style
    code_style = ParagraphStyle(
        "CustomCode",
        parent=base_styles["Code"],
        fontSize=8,
        leading=10,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.HexColor("#f2f2f2"),
        borderColor=colors.HexColor("#dddddd"),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=5,
        spaceAfter=8
    )

    # Return all styles in one dictionary
    return {
        "title": title_style,
        "heading": heading_style,
        "subheading": subheading_style,
        "normal": normal_style,
        "bullet": bullet_style,
        "code": code_style
    }


def add_page_number(canvas, document):
    """
    Adds page numbers to the PDF.

    Parameters:
        canvas: ReportLab canvas object
        document: ReportLab document object
    """

    # Get current page number
    page_number = canvas.getPageNumber()

    # Create footer text
    text = f"Page {page_number}"

    # Set font size for footer
    canvas.setFont("Helvetica", 8)

    # Draw page number at bottom-right
    canvas.drawRightString(
        A4[0] - 2 * cm,
        1.2 * cm,
        text
    )


def markdown_table_to_reportlab(table_lines):
    """
    Converts a simple markdown table into a ReportLab table.

    Parameters:
        table_lines: list of markdown table lines

    Returns:
        ReportLab Table object
    """

    # This list will contain table rows
    rows = []

    # Loop through markdown table lines
    for line in table_lines:

        # Skip separator lines like |---|---|
        if "---" in line:
            continue

        # Remove starting and ending pipe, then split columns
        columns = line.strip().strip("|").split("|")

        # Clean spaces around each column
        columns = [clean_markdown_line(col.strip()) for col in columns]

        # Add row to table
        rows.append(columns)

    # If there are no valid rows, return None
    if len(rows) == 0:
        return None

    # Create ReportLab table
    table = Table(rows, hAlign="LEFT")

    # Apply table style
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    return table


def convert_markdown_to_pdf(markdown_path, pdf_path):
    """
    Converts a markdown document into a PDF.

    Parameters:
        markdown_path: input markdown file path
        pdf_path: output PDF file path
    """

    # Convert paths to Path objects
    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path)

    # Check if markdown file exists
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    # Create output folder if it does not exist
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Create custom PDF styles
    styles = create_styles()

    # Create PDF document
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.7 * cm
    )

    # This list will contain all PDF elements
    story = []

    # Read markdown lines
    lines = markdown_path.read_text(encoding="utf-8").splitlines()

    # Variable used to detect code blocks
    inside_code_block = False

    # Temporary list for code block lines
    code_lines = []

    # Temporary list for markdown table lines
    table_lines = []

    # Loop through every line in the markdown file
    for line in lines:

        # Clean line spacing
        stripped_line = line.strip()

        # Detect code block start or end
        if stripped_line.startswith("```"):

            # If we are already inside a code block, this closes it
            if inside_code_block:

                # Join code lines into one text block
                code_text = "<br/>".join(code_lines)

                # Add code block to story
                story.append(Paragraph(code_text, styles["code"]))

                # Add small space after code block
                story.append(Spacer(1, 6))

                # Reset code block variables
                code_lines = []
                inside_code_block = False

            else:
                # Start a new code block
                inside_code_block = True
                code_lines = []

            # Continue to next line
            continue

        # If we are inside a code block, collect lines without parsing them
        if inside_code_block:
            code_lines.append(clean_markdown_line(stripped_line))
            continue

        # Detect markdown table lines
        if stripped_line.startswith("|") and stripped_line.endswith("|"):
            table_lines.append(stripped_line)
            continue

        # If a table just ended, render it before processing normal content
        if len(table_lines) > 0:
            table = markdown_table_to_reportlab(table_lines)

            if table is not None:
                story.append(table)
                story.append(Spacer(1, 8))

            table_lines = []

        # Skip empty lines but add small spacing
        if stripped_line == "":
            story.append(Spacer(1, 4))
            continue

        # Skip horizontal rules
        if stripped_line == "---":
            story.append(Spacer(1, 6))
            continue

        # Main title
        if stripped_line.startswith("# "):
            text = clean_markdown_line(stripped_line.replace("# ", "", 1))
            story.append(Paragraph(text, styles["title"]))
            continue

        # Section heading
        if stripped_line.startswith("## "):
            text = clean_markdown_line(stripped_line.replace("## ", "", 1))
            story.append(Paragraph(text, styles["heading"]))
            continue

        # Subsection heading
        if stripped_line.startswith("### "):
            text = clean_markdown_line(stripped_line.replace("### ", "", 1))
            story.append(Paragraph(text, styles["subheading"]))
            continue

        # Bullet list
        if stripped_line.startswith("- "):
            text = clean_markdown_line(stripped_line.replace("- ", "• ", 1))
            story.append(Paragraph(text, styles["bullet"]))
            continue

        # Numbered list
        if stripped_line[0:2].isdigit() and ". " in stripped_line[:4]:
            text = clean_markdown_line(stripped_line)
            story.append(Paragraph(text, styles["normal"]))
            continue

        # Normal paragraph
        text = clean_markdown_line(stripped_line)
        story.append(Paragraph(text, styles["normal"]))

    # If the file ends while a table is still collected, render it
    if len(table_lines) > 0:
        table = markdown_table_to_reportlab(table_lines)

        if table is not None:
            story.append(table)
            story.append(Spacer(1, 8))

    # Build PDF
    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    print(f"PDF created successfully: {pdf_path}")


def main():
    """
    Main function used to generate the Technical Analysis PDF.
    """

    # Define input markdown path
    markdown_path = Path("docs/technical_analysis.md")

    # Define output PDF path
    pdf_path = Path("docs/technical_analysis.pdf")

    # Convert markdown to PDF
    convert_markdown_to_pdf(
        markdown_path=markdown_path,
        pdf_path=pdf_path
    )


# This block runs only when the script is executed directly
if __name__ == "__main__":
    main()