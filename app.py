pdf_text = extract_pdf_text(
    uploaded_pdf
)

st.write("Text Length:", len(pdf_text))

st.text(pdf_text[:1000])

chunks = chunk_text(
    pdf_text
)

st.write("Chunks Created:", len(chunks))