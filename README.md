# BedrockChat - Final Project

A final implementation of Bedrock chat interface.
A streamlit based chat interface for Amazon Bedrock foundation models with a Streamlit frontend and Flask API backend. Currently supports Claude, Nova, Mistral, and Pixtral models with user management and usage logging. Supports RAG based knowledge for Neuroscience materials and user-uploaded documents.

## [Live App](https://bedrock-chat-1.onrender.com/)

## Goals and Use Cases

### Project Goals
This project demonstrates the development of a production-ready customizable RAG (Retrieval Augmented Generation) system that showcases:

1. **Multi-Model AI Integration**: Seamless integration with AWS Bedrock foundation models (Claude-2, Nova, Mistral)
2. **RAG Implementation**: Document processing and semantic search capabilities
5. **Production Deployment**: Cloud-ready with AWS App Runner deployment

### Primary Use Cases

#### 1. **Research Assistant**
- **Functionality**: Query pre-loaded neuroscience materials (Brain Facts Book, Neuroscience: Science of the Brain)

#### 2. **Personal Knowledge Management**
- **Functionality**: Upload personal PDF/TXT documents for personalized RAG system
able document processing with comprehensive logging

#### 3. **AI Model Comparison Platform**
- **Functionality**: Test different foundation models (Claude, Nova, Mistral) on same queries


### What I learned

Throughout this project, I gained hands-on experience with:

**AWS Cloud Infrastructure:**
- AWS Bedrock for foundation model APIs (Claude, Nova, Mistral)
- S3 for document storage and user file management
- App Runner for containerized deployments
- IAM roles and security policies

**Production-Level Development:**
- Building scalable RAG (Retrieval-Augmented Generation) systems
- Implementing user authentication and API key management
- Content filtering with AWS Bedrock Guardrails

**System Architecture & Integration:**
- Designing microservices with Flask API backend
- Creating responsive frontends with Streamlit
- Vector database operations with Pinecone
- Seamless module integration and dependency management

**DevOps & Deployment:**
- Automated deployments with Render.com
- Environment variable management
- Git workflow and version control
- Comprehensive testing with pytest

**Security & Best Practices:**
- API authentication and authorization
- Input/output validation and sanitization
- Secure credential management

**Documentation & Monitoring:**
- API documentation with Swagger/OpenAPI
- Production monitoring and debugging

### [Additional Learning (Week 2)](https://github.com/owais-kamdar/mistral-chat)

In parallel with this project, I explored local AI model deployment through a separate repository:

**Local Model Experimentation:**
- Setting up and running Mistral 7B locally without external dependencies
- Command-line interface development for AI interactions
- Understanding the differences between local vs. hosted model architectures

This comparative experience helped deepen my understanding of the trade-offs between local and cloud-based AI solutions.


## Project Structure

```
bedrock/
├── src/                   
│   ├── core/              
│   │   ├── bedrock.py     
│   │   ├── rag.py         
│   │   ├── vector_store.py 
│   │   ├── initialize_rag.py 
│   │   ├── config.py     
│   │   └── README.md     
│   ├── app/              
│   │   ├── flaskapp.py   
│   │   └── README.md     
│   ├── ui/               
│   │   ├── streamlit_app.py 
│   │   ├── dashboard.py   
│   │   └── README.md      
│   ├── utils/             
│   │   ├── logger.py      
│   │   ├── user_manager.py 
│   │   └── README.md      
│   └── README.md          
├── tests/                 
│   ├── test_bedrock.py    
│   └── README.md          
├── config/                
│   ├── .env               
│   └── README.md          
├── logs/                  
├── run_api.py             # Flask API entry point
├── run_streamlit.py       # Streamlit UI entry point
├── run_dash.py            # Dashboard entry point
├── run_initialize.py      # RAG system initialization entry point
├── requirements.txt       
└── README.md              
```

## Current Features

- **Multiple Bedrock Models**:
  - Claude-2
  - Nova
  - Mistral
- **Guardrail Support**:
  - Prompt and response filters
  - Profanity and topic blocking
  - PII Blocks/Masks
- **User Management**:
  - Sequential user IDs (user-1, user-2, etc.)
  - API key generation and validation
  - User-specific document storage
- **Comprehensive Dashboard**:
  - Usage statistics
  - User filtering
  - Token count, response time, conversation history, etc
- **RAG (Retrieval Augmented Generation)**:
  - PDF document indexing
  - Semantic search with Pinecone DB
  - Multiple context sources (Neuroscience Guide, User Documents, Both)
- **File Upload System**:
  - Upload PDF and TXT files only
  - Automatic document processing and indexing
  - User-specific document management
  - Personal knowledge base creation
- **Test Suite**:
  - Core module coverage: 
    - `initialize_rag.py`: 97%  
    - `config.py`: 86%  
    - `logger.py`: 85%
    - `user_manager.py`: 75%
    - `bedrock.py`: 70%
    - `vector_store.py`: 70%
    - `rag.py`: 67%
    - `flaskapp.py`: 59%

    ### - Average Coverage: 76%

## Setup

1. Clone the repository:
```bash
git clone https://github.com/owais-kamdar/bedrock-chat.git
cd bedrock
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set up configuration:
   - See [`config/README.md`](config/README.md) for detailed configuration options and environment variables

## Usage

1. Initialize the RAG system (first time only):
```bash
python run_initialize.py
```

2. Start the Flask API:
```bash
python run_api.py
```

3. In a separate terminal, start the Streamlit interface:
```bash
python run_streamlit.py
```

4. To access the dashboard:
```bash
python run_dash.py
```


## User Management

### User IDs
- Sequential format: `user-1`, `user-2`, `user-3`, etc.
- Automatically generated for new users

### API Keys
- One API key per user


## File Upload

### Uploading Documents
1. In the Streamlit sidebar, use the file uploader to select PDF or TXT files
2. Click "Upload & Index File" to process and index your document
3. Your document will be automatically chunked, embedded, and stored in your personal namespace

### Managing Your Documents
- View all uploaded files in the "Your Documents" section
- See file details including size and upload date
- Delete files you no longer need
- All documents are stored securely in S3 with user-specific namespaces

### Using Your Documents
- Select Your choice for context in the Context Source setting
- The system will search your uploaded documents for relevant context


## API Endpoints

### User Management
- `POST /api-keys` - Create API key for a user
- `GET /health` - Health check endpoint

### Chat Interface
- `POST /chat/{user_id}` - Send message to AI model
  - Use `user_id="new"` to generate a new user
  - `context_source`: "Neuroscience Guide", "Your Documents", "Both", "None"

### File Management
- `POST /upload/{user_id}` - Upload documents for a user
- `GET /files/{user_id}` - List user uploaded files
- `DELETE /files/{user_id}` - Delete a user uploaded file


## Testing

### Run Test Suite
```bash
# Run all tests with coverage
python -m pytest tests/test_bedrock.py --cov=src --cov-report=term-missing -v
```

## Development Timeline

### Week 1: Foundation Setup
**Initial Bedrock System with Model Calls through Streamlit Interface**
- Basic Streamlit interface for AWS Bedrock integration
- Initial Claude-2 model support
- Simple chat functionality


### Week 2: Different Project -- Seperate Repo
**Separate Local Mistral Chat with Dashboard**
- Mistral model locally 
- Initial dashboard concepts for usage tracking


### Week 3: API Integration & Security
**Additional Models Support with Flask App Integration**
- Flask API backend development (`flaskapp.py`)
- Session management and logging infrastructure
- Guardrails implementation through AWS Bedrock

### Week 4: RAG System Implementation
**RAG System using S3 and Pinecone**
- **Document Processing**: Text chunking 
- **Vector Store**: Pinecone DB for Storage
- **Embeddings**: Automated document chunking and embedding generation
- **Additional Security**: Enhanced guardrail filters for content safety

### Week 5: Analytics & Centralized Storage
**Dashboard System and S3 Integration**
- **Analytics Dashboard**: Session-based analytics with detailed metrics and user filtering
- **Data Visualization**: Response time tracking, token usage metrics, conversation history
- **AWS S3 Storage**: Centralized storage architecture with organized folder structure (`/logs/`, `/rag/`, `/api_keys/`)

### Week 6: Personalization & Deployment
**User Upload Integration and Production Readiness**
- **Personal RAG**: User-specific document upload and indexing system
- **Production Features**: Error handling, rate limiting, logging
- **Documentation**: Thorough documentation and setup
- **Deployment**:  Using Render


## License

This project is licensed under the MIT License.
