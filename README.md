---
app_file: app.py
colorFrom: indigo
colorTo: teal
emoji: 📄
pinned: false
sdk: streamlit
title: DocuMind
---

# 📄 DocuMind --- AI-Powered Document Question Answering Assistant

DocuMind is a Streamlit-based application that lets users upload a PDF,
ask questions about its contents, and receive answers grounded in the
uploaded document. It uses local Hugging Face models, semantic search,
and extractive question answering. Answers include a source page and a
supporting passage so users can check where the information came from.

> **Important:** DocuMind is designed to answer from the selected PDF,
> not from general internet knowledge. If the answer cannot be found
> confidently, the app displays:
> `Information not found in provided document`.

## ✨ Features

-   **PDF upload:** Upload a PDF through the sidebar.
-   **Recent documents:** Reopen PDFs previously uploaded during the
    current app's available storage lifetime.
-   **Text extraction:** Extract readable text page by page from a PDF.
-   **Text chunking:** Split extracted text into smaller overlapping
    sections for retrieval.
-   **Semantic search:** Find passages relevant to the user's question
    using sentence embeddings and FAISS.
-   **Extractive question answering:** Identify an answer span from
    retrieved passages with a Hugging Face QA model.
-   **Source references:** Show the source page and supporting passage
    for a found answer.
-   **Retrieved passage inspection:** View the passages considered by
    the retrieval stage.
-   **Question history:** Review earlier questions and answers for the
    active document.
-   **Local model inference:** No paid LLM API key is required.

## 🧠 How It Works

1.  **Upload:** The user selects a PDF file.
2.  **Extract:** PyMuPDF reads text from each page and keeps the page
    number.
3.  **Chunk:** NLTK sentence tokenization and the text-processing module
    divide page text into manageable chunks. A small sentence overlap
    helps preserve context between adjacent chunks.
4.  **Embed:** `sentence-transformers/all-MiniLM-L6-v2` converts chunks
    into vector embeddings.
5.  **Index:** FAISS stores the vectors and supports similarity search.
6.  **Retrieve:** The application retrieves the top relevant chunks for
    the question.
7.  **Answer:** `deepset/tinyroberta-squad2` performs extractive
    question answering over those chunks.
8.  **Show evidence:** The UI displays the answer, source page, and
    supporting passage. If no usable answer is returned, the not-found
    message is shown.

### Models and main libraries

  ------------------------------------------------------------------------------------------
  Component               Technology                                 Purpose
  ----------------------- ------------------------------------------ -----------------------
  User interface          Streamlit                                  Interactive web
                                                                     application

  PDF text extraction     PyMuPDF                                    Read text and page
                                                                     numbers

  Text processing         NLTK                                       Sentence tokenization

  Chunk representation    LangChain `Document`                       Store chunk text and
                                                                     metadata

  Embeddings              `sentence-transformers/all-MiniLM-L6-v2`   Convert text into
                                                                     vectors

  Vector search           FAISS                                      Retrieve semantically
                                                                     similar chunks

  Question answering      `deepset/tinyroberta-squad2`               Extract answer spans
                                                                     from context

  Model framework         Hugging Face Transformers                  Load and run the QA
                                                                     model
  ------------------------------------------------------------------------------------------

## 🗂️ Project Structure

``` text
Document Q&A Assistant/
│
├── app.py                    # Streamlit user interface and application flow
├── document_reader.py        # PDF text extraction
├── text_processor.py         # Sentence splitting and chunk creation
├── retriever.py              # Embedding model, FAISS index and retrieval
├── answer_generator.py       # Extractive question-answering logic
│
├── sample_documents/
│   └── sample.pdf            # Sample PDF for testing (if included)
│
├── screenshots/              # Optional project screenshots
├── uploaded_documents/       # Runtime uploads; excluded from Git
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files and folders excluded from Git
```

## ⚙️ Installation and Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd "Document Q&A Assistant"
```

Replace the repository URL with your own GitHub repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

``` powershell
.\venv\Scripts\activate.bat
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use Python 3.10 or 3.11 for a compatible setup. The project uses the
Transformers v4 question-answering pipeline, so keep Transformers below
version 5.

### 4. Run the application

``` powershell
streamlit run app.py
```

Streamlit will print a local URL, usually:

``` text
http://localhost:8501
```

Open that address in your browser.

## 💻 Recommended `requirements.txt`

Make sure the repository's `requirements.txt` includes the dependencies
used by the project and pins Transformers below v5:

``` text
streamlit
pymupdf
nltk
langchain
langchain-core
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
transformers<5
torch
```

For a reproducible submission, you can later freeze and review the exact
versions from your working environment. Avoid blindly adding every
package installed on your computer.

## 🚀 Deploy on Hugging Face Spaces

This project is configured for a **Streamlit Space**.

1.  Create a new Space on Hugging Face.
2.  Select **Streamlit** as the SDK.
3.  Upload or push the project files to the Space repository.
4.  Ensure the root `README.md` begins with the YAML Space configuration
    shown at the top of this file.
5.  Ensure `requirements.txt` contains `transformers<5`.
6.  Wait for the Space to install dependencies and build the
    application.
7.  Open the Space URL and test it with a PDF.

### Deployment notes

-   The embedding and QA models are downloaded from Hugging Face when
    first required. Initial startup or first use can take time.
-   Free CPU resources may be slower for model loading and inference.
-   Do not commit your virtual environment, Hugging Face cache, or
    private documents.
-   Files uploaded to a hosted Space may be stored on temporary local
    storage. Do not assume user uploads will survive a restart or
    rebuild unless persistent storage is configured.
-   Large PDFs and scanned/image-only PDFs may take longer or may not
    work if they contain no extractable text.

## 🧪 Testing the Application

Use a PDF whose contents you are allowed to test. The repository's
sample PDF, if included, is titled **Farm-to-Market Haryana --- Complete
Solution Blueprint**.

Example questions for that sample document:

  ----------------------------------------------------------------------------------------------
  Test                    Example question        Expected behavior
  ----------------------- ----------------------- ----------------------------------------------
  Relevant question       What is the main        Return an answer based on the PDF and show its
                          objective of the        source page/passage.
                          platform?               

  Relevant question       What types of           Retrieve the relevant passage and provide an
                          information does the    extractive answer.
                          platform combine?       

  Relevant question       Why is transport        Answer only if the PDF contains supporting
                          information useful to   information.
                          the platform?           

  Unsupported question    What is the capital of  Display
                          Japan?                  `Information not found in provided document`
                                                  if the PDF does not contain the answer.
  ----------------------------------------------------------------------------------------------

### Example interaction format

``` text
User: What is the main objective of the platform?

DocuMind: [An answer extracted from the relevant PDF passage]

Source page: [Page number shown by the application]

Supporting passage: [Relevant text from the uploaded PDF]
```

The answer wording and page number depend on the actual PDF and model
output. Use the running application to capture genuine screenshots or
transcripts for your project submission.

## 📌 Current Scope and Limitations

-   **PDF input:** The current application accepts PDF files. TXT upload
    is not included in this version.
-   **Readable text required:** Scanned PDFs without an embedded text
    layer may not produce readable content. OCR is not currently
    included.
-   **Extractive answers:** The QA model selects an answer span from
    retrieved text; it does not write long, free-form summaries.
-   **Retrieval quality:** Semantic retrieval can miss relevant
    passages, especially in very long, poorly formatted, or unusual
    documents.
-   **Not-found behavior:** The app has a fallback message for questions
    where it cannot produce a usable answer. Model confidence scores are
    not a guarantee of correctness.
-   **Storage:** Recent uploads are saved in the app's
    `uploaded_documents` directory. On hosted platforms, local storage
    may be temporary.
-   **Language:** The selected embedding and QA models are primarily
    intended for English text. Results may be weaker for other
    languages.

Always verify important information against the source document.

## 🔐 Privacy and Security

-   Uploaded documents are processed by the running application.
-   Do not upload confidential, sensitive, or personal documents to a
    public demo unless you understand the hosting platform's privacy and
    storage settings.
-   The `uploaded_documents/` folder is excluded from Git by
    `.gitignore`.
-   Never commit passwords, API keys, access tokens, or private
    documents.

## 🔮 Future Enhancements

-   TXT and DOCX document support.
-   OCR for scanned PDFs.
-   Better retrieval options such as BM25 and hybrid search.
-   Improved answer-grounding checks and retrieval thresholds.
-   Multi-document question answering.
-   Downloadable chat history and source references.
-   Persistent document storage for hosted deployments.
-   More multilingual embedding and QA model support.

## 👩‍💻 Author

**Neha**

