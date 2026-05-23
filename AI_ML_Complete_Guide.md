# AI & ML — Complete Concepts Guide (Basics → Enterprise)

---

## PART 1: FOUNDATIONS

### 1.1 Types of AI
| Type | Description | Example |
|------|-------------|---------|
| Narrow AI (ANI) | Ek specific task ke liye | Chess engine, spam filter |
| General AI (AGI) | Human-level reasoning | (Not yet achieved) |
| Super AI (ASI) | Human se zyada intelligent | (Theoretical) |

### 1.2 Types of Machine Learning

```
Machine Learning
├── Supervised Learning      → labeled data se sikhna
│   ├── Classification       → category predict karna (spam/not spam)
│   └── Regression           → number predict karna (house price)
│
├── Unsupervised Learning    → unlabeled data se patterns dhundhna
│   ├── Clustering           → groups banana (customer segments)
│   └── Dimensionality Reduction → data compress karna (PCA)
│
├── Semi-Supervised          → thoda labeled + bahut saara unlabeled
│
└── Reinforcement Learning   → reward/penalty se sikhna (game AI)
```

### 1.3 Core Terminology
- **Feature** — input variable (age, salary)
- **Label/Target** — output variable (churn: yes/no)
- **Training Data** — jis data se model seekhta hai
- **Validation Data** — hyperparameter tuning ke liye
- **Test Data** — final evaluation ke liye
- **Overfitting** — training par bahut accha, test par kharab
- **Underfitting** — dono par kharab (model too simple)
- **Bias-Variance Tradeoff** — simplicity vs complexity balance
- **Cross-Validation** — k-fold splitting for robust evaluation
- **Hyperparameter** — model training se pehle set karte hain (learning rate, depth)

---

## PART 2: CLASSICAL ML ALGORITHMS

### 2.1 Supervised — Classification
| Algorithm | Kab Use Karein | Key Concept |
|-----------|---------------|-------------|
| Logistic Regression | Binary classification, interpretable | Sigmoid function, log-odds |
| Decision Tree | Simple rules, interpretable | Gini / Entropy splitting |
| Random Forest | Tabular data, robust | Bagging + multiple trees |
| XGBoost | Competitions, tabular data | Boosting (sequential trees) |
| LightGBM | Large datasets, fast | Leaf-wise growth |
| SVM | High-dimensional, small data | Hyperplane + kernel trick |
| KNN | Simple baseline | Nearest neighbor distance |
| Naive Bayes | Text classification, fast | Bayes theorem + independence |

### 2.2 Supervised — Regression
- **Linear Regression** — straight line fit, OLS
- **Ridge / Lasso** — regularization (L2 / L1) for overfitting control
- **ElasticNet** — L1 + L2 combined
- **Polynomial Regression** — non-linear relationships
- **SVR** — SVM for regression

### 2.3 Unsupervised
| Algorithm | Type | Use Case |
|-----------|------|----------|
| K-Means | Clustering | Customer segmentation |
| DBSCAN | Clustering | Anomaly detection, arbitrary shapes |
| Hierarchical | Clustering | Dendrograms, biology |
| PCA | Dim. Reduction | Visualization, noise removal |
| t-SNE | Dim. Reduction | 2D/3D visualization |
| UMAP | Dim. Reduction | Faster t-SNE alternative |
| Autoencoders | Dim. Reduction | Feature learning, compression |

### 2.4 Evaluation Metrics
```
Classification:         Regression:
- Accuracy              - MAE (Mean Absolute Error)
- Precision             - MSE / RMSE
- Recall                - R² Score
- F1 Score              - MAPE
- AUC-ROC               - Huber Loss
- Confusion Matrix      
- Log Loss              
```

---

## PART 3: DEEP LEARNING

### 3.1 Neural Network Basics
```
Input Layer → Hidden Layers → Output Layer
     ↑
 Weights + Biases
 Activation Functions: ReLU, Sigmoid, Tanh, GELU, Softmax
 Loss Functions: Cross-Entropy, MSE, Huber
 Optimizers: SGD, Adam, AdamW, RMSprop
 Regularization: Dropout, Batch Norm, Layer Norm, Weight Decay
```

### 3.2 Key Architectures

#### CNN (Convolutional Neural Network)
- **Use**: Images, video, spatial data
- **Components**: Conv layers → Pooling → Flatten → FC layers
- **Famous Models**: ResNet, VGG, EfficientNet, MobileNet

#### RNN / LSTM / GRU
- **Use**: Sequential data, time series, text
- **RNN Problem**: Vanishing gradient
- **LSTM Fix**: Cell state + gates (forget, input, output)
- **GRU**: Simpler LSTM, fewer parameters

#### Transformer
- **2017 — "Attention Is All You Need"**
- **Key Concept**: Self-Attention mechanism
- **Formula**: `Attention(Q, K, V) = softmax(QKᵀ/√d_k) × V`
- **Components**:
  - Multi-Head Attention
  - Positional Encoding
  - Feed Forward Network
  - Layer Normalization
  - Encoder-Decoder architecture

### 3.3 Training Techniques
- **Transfer Learning** — pretrained model fine-tune karna
- **Data Augmentation** — artificial training data banana
- **Early Stopping** — validation loss badhe to ruk jao
- **Learning Rate Scheduling** — cosine annealing, warmup
- **Gradient Clipping** — exploding gradients rokna
- **Mixed Precision Training** — FP16 + FP32 (faster, less memory)

---

## PART 4: NLP (Natural Language Processing)

### 4.1 Text Processing Pipeline
```
Raw Text
  → Tokenization (word/subword/character)
  → Normalization (lowercase, punctuation)
  → Stop Word Removal
  → Stemming / Lemmatization
  → Embeddings (convert to vectors)
```

### 4.2 Word Embeddings Evolution
| Model | Year | Key Idea |
|-------|------|----------|
| Bag of Words (BoW) | Classic | Word counts, no order |
| TF-IDF | Classic | Term frequency + inverse doc freq |
| Word2Vec | 2013 | CBOW + Skip-gram, context matters |
| GloVe | 2014 | Global co-occurrence statistics |
| FastText | 2016 | Subword embeddings |
| ELMo | 2018 | Contextual embeddings (BiLSTM) |
| BERT | 2018 | Bidirectional Transformer |
| GPT | 2018 | Autoregressive Transformer |

### 4.3 BERT vs GPT
| | BERT | GPT |
|---|------|-----|
| Direction | Bidirectional | Left-to-right |
| Task | Understanding (classification) | Generation (text) |
| Training | Masked LM + Next Sentence Pred | Causal LM |
| Use | NER, QA, classification | Chat, completion, coding |

### 4.4 NLP Tasks
- **NER** — Named Entity Recognition (person, place, org)
- **POS Tagging** — Part of Speech
- **Sentiment Analysis** — positive/negative/neutral
- **Text Classification** — spam, category
- **Machine Translation** — English → Hindi
- **Summarization** — extractive vs abstractive
- **Question Answering** — span extraction / generative
- **Text Generation** — GPT-style completion
- **Coreference Resolution** — "he/she" kisko refer kar raha hai

---

## PART 5: LARGE LANGUAGE MODELS (LLMs)

### 5.1 LLM Architecture
```
Pre-training Phase:
  Massive text corpus (internet, books, code)
  → Self-supervised learning (next token prediction)
  → Base model (raw knowledge, no alignment)

Post-training Phase:
  → SFT (Supervised Fine-Tuning on instruction data)
  → RLHF (Reinforcement Learning from Human Feedback)
  → DPO (Direct Preference Optimization) — RLHF alternative
  → Final aligned model (ChatGPT, Claude, Gemini)
```

### 5.2 Key LLM Concepts

#### Tokens & Context Window
- Token ≈ 0.75 words (English)
- Context window = max tokens model dekh sakta hai
- GPT-4: 128K | Claude: 200K | Gemini: 1M+

#### Temperature & Sampling
- **Temperature 0** — deterministic (same answer always)
- **Temperature 1** — balanced creativity
- **Temperature >1** — very random
- **Top-p (nucleus sampling)** — top probability tokens se sample
- **Top-k** — top k tokens se sample

#### Attention Types
- **Full Attention** — O(n²) memory
- **Flash Attention** — memory-efficient, same result
- **Grouped Query Attention (GQA)** — fewer KV heads (Llama 2/3)
- **Sliding Window Attention** — local context only (Mistral)

### 5.3 Prompting Techniques
| Technique | Description | Use When |
|-----------|-------------|----------|
| Zero-shot | Koi example nahi | Simple tasks |
| Few-shot | 2-5 examples dena | Format/style control |
| Chain of Thought (CoT) | "Think step by step" | Math, reasoning |
| ReAct | Reason + Act + Observe | Agent tasks |
| Tree of Thought (ToT) | Multiple reasoning paths | Complex problems |
| Self-Consistency | Multiple answers, majority vote | Accuracy critical |
| Constitutional AI | Rules-based self-critique | Safety |
| Role Prompting | "You are an expert in..." | Domain-specific |

### 5.4 Fine-tuning Techniques
| Method | When | Cost |
|--------|------|------|
| Full Fine-tuning | Lots of data, lots of compute | Very High |
| LoRA | Parameter-efficient, popular | Low |
| QLoRA | LoRA + 4-bit quantization | Very Low |
| Adapter Layers | Plug-in modules | Medium |
| Prefix Tuning | Trainable prefix tokens | Low |
| Prompt Tuning | Only prompt tokens train | Very Low |

### 5.5 Inference Optimization
- **Quantization** — FP32 → INT8/INT4 (smaller, faster)
- **Pruning** — unnecessary weights remove karna
- **Knowledge Distillation** — large model → small model
- **Speculative Decoding** — small model draft, large verify
- **KV Cache** — attention cache reuse
- **Continuous Batching** — dynamic batch sizes for throughput
- **vLLM** — PagedAttention for efficient serving

---

## PART 6: RAG (Retrieval Augmented Generation)

### 6.1 Why RAG?
- LLMs ka knowledge cutoff hota hai
- Hallucination reduce karna
- Private/company data use karna
- Real-time information chahiye

### 6.2 Basic RAG Pipeline
```
User Query
  → Embed query (vector)
  → Vector DB search (ANN search)
  → Top-K relevant chunks retrieve
  → Inject into LLM prompt as context
  → LLM generates grounded answer
```

### 6.3 Advanced RAG Techniques
| Technique | What it Does |
|-----------|-------------|
| **Hybrid Search** | Vector search + BM25 keyword search combine |
| **Reranking** | Cross-encoder se chunks reorder (Cohere Rerank) |
| **Query Rewriting** | HyDE — hypothetical doc embed karo |
| **Multi-query** | Multiple query versions se retrieve |
| **Parent-Child Chunking** | Small chunks retrieve, large chunks pass to LLM |
| **Contextual Compression** | Irrelevant parts filter karo before LLM |
| **RAPTOR** | Hierarchical summarization + retrieval |
| **Self-RAG** | Model decide kare kab retrieve karna hai |
| **CRAG** | Corrective RAG — low quality docs ko web search se replace |

### 6.4 Chunking Strategies
- **Fixed-size** — 512 tokens, 50 token overlap
- **Sentence-based** — sentence boundaries respect karo
- **Semantic** — meaning-based splitting
- **Document-structure** — headers/sections follow karo
- **Recursive** — hierarchy-aware splitting

### 6.5 Vector Databases
| DB | Best For | Notes |
|----|---------|-------|
| **Pinecone** | Production, managed | Fully hosted |
| **Weaviate** | Hybrid search, GraphQL | Self-host or cloud |
| **Chroma** | Local dev, simple | In-memory or persistent |
| **Qdrant** | High performance, Rust | Self-host |
| **pgvector** | Already using Postgres | Extension |
| **FAISS** | Research, local | Meta's library |
| **Milvus** | Billion-scale | Self-host |

---

## PART 7: AI AGENTS

### 7.1 What is an Agent?
> LLM + Tools + Memory + Planning = Agent

Agent ek autonomous system hai jo:
- User ka goal samjhe
- Tools use kare (web search, code execution, APIs)
- Multi-step planning kare
- Feedback loop mein kaam kare

### 7.2 Agent Patterns

#### ReAct (Reason + Act)
```
Thought: Mujhe X karna hai
Action: tool_name("input")
Observation: tool result
Thought: Ab Y karna hai
Action: ...
Final Answer: ...
```

#### Plan & Execute
```
Planner → [Step 1, Step 2, Step 3, ...]
Executor → Step 1 execute → result
         → Step 2 execute → result
         → Final Answer
```

#### Reflection / Self-Critique
```
Agent answer generate kare
  → Critic model evaluate kare
  → Improvement suggestions
  → Agent revise kare
  → Final answer
```

#### Multi-Agent System
```
Supervisor Agent
├── Research Agent     → web search, data gathering
├── Analyst Agent      → data analysis, reasoning
├── Writer Agent       → content generation
└── Reviewer Agent     → quality check
```

### 7.3 Agent Frameworks
| Framework | Language | Best For |
|-----------|----------|----------|
| **LangGraph** | Python | Stateful, cyclical workflows |
| **LangChain** | Python | Chain-based, many integrations |
| **AutoGen** | Python | Multi-agent conversations |
| **CrewAI** | Python | Role-based agent teams |
| **LlamaIndex** | Python | RAG-focused agentic systems |
| **Semantic Kernel** | C#/Python | Microsoft ecosystem |
| **Swarm** | Python | OpenAI's lightweight multi-agent |

### 7.4 Tool Types for Agents
- **Code Interpreter** — Python execute karo
- **Web Search** — DuckDuckGo, Tavily, SerpAPI
- **Calculator** — math operations
- **File System** — read/write files
- **Database** — SQL queries
- **APIs** — REST, GraphQL calls
- **Email/Calendar** — SMTP, Google Calendar
- **Browser** — web scraping, form filling

### 7.5 Memory Types in Agents
```
Short-term (In-context):   current conversation
Working Memory:             tool results, intermediate state
Episodic Memory:            past conversation summaries
Semantic Memory:            facts, preferences (vector DB)
Procedural Memory:          how to do things (few-shot examples)
```

---

## PART 8: COMPUTER VISION

### 8.1 Core Tasks
| Task | Description | Model |
|------|-------------|-------|
| Classification | Image ka label predict | ResNet, EfficientNet, ViT |
| Object Detection | Bounding boxes + labels | YOLOv8, DETR, Faster R-CNN |
| Segmentation | Pixel-level classification | SAM, Mask R-CNN, DeepLab |
| Pose Estimation | Body keypoints | MediaPipe, OpenPose |
| OCR | Text extraction from images | Tesseract, PaddleOCR, TrOCR |
| Face Recognition | Identity matching | FaceNet, ArcFace |
| Depth Estimation | 3D depth from 2D | DPT, MiDaS |

### 8.2 Generative Models
| Model | Type | Output |
|-------|------|--------|
| GAN (2014) | Generator + Discriminator | Realistic images |
| VAE | Variational Autoencoder | Smooth latent space |
| Diffusion Models | Noise → Image via denoising | High quality images |
| Stable Diffusion | Latent Diffusion | Text-to-image |
| DALL-E 3 | OpenAI | Text-to-image |
| Midjourney | Commercial | Artistic images |
| Sora | OpenAI | Text-to-video |

---

## PART 9: REINFORCEMENT LEARNING

### 9.1 Core Concepts
```
Agent → Action → Environment → Reward + Next State → Agent
                      ↑
                   Policy π: State → Action
                   Value function V(s): expected future reward
                   Q-function Q(s,a): action value
```

### 9.2 Algorithms
| Algorithm | Type | Use |
|-----------|------|-----|
| Q-Learning | Model-free, tabular | Simple environments |
| DQN | Deep Q-Network | Atari games |
| PPO | Policy gradient | Most common, stable |
| A3C | Actor-Critic | Parallel training |
| SAC | Off-policy, continuous | Robotics |
| TD3 | Deterministic policy | Continuous control |

### 9.3 RLHF (RL from Human Feedback)
```
1. Supervised Fine-Tuning (SFT)
   → Human-written examples se fine-tune

2. Reward Model Training
   → Human rankers: "Answer A > Answer B"
   → Train reward model to predict human preference

3. PPO Optimization
   → LLM ko reward maximize karne ke liye train karo
   → KL divergence penalty (original model se door mat jao)

Result: ChatGPT, Claude, Gemini
```

### 9.4 RLHF Alternatives
- **DPO (Direct Preference Optimization)** — no reward model needed
- **ORPO** — simpler, single-stage
- **KTO** — binary feedback (thumbs up/down)

---

## PART 10: MLOPS & PRODUCTION

### 10.1 ML Lifecycle
```
Problem Definition
  → Data Collection & Annotation
  → EDA (Exploratory Data Analysis)
  → Feature Engineering
  → Model Development (experiments)
  → Model Evaluation & Selection
  → Model Deployment
  → Monitoring & Maintenance
  → Retraining (when drift detected)
```

### 10.2 Feature Engineering & Stores
- **Feature Engineering**: raw data → useful features
  - Encoding (One-hot, Label, Target encoding)
  - Scaling (StandardScaler, MinMax, RobustScaler)
  - Imputation (mean, median, KNN, model-based)
  - Feature Selection (correlation, mutual info, RFE)
- **Feature Store**: Feast, Tecton, Hopsworks
  - Online store (low latency, serving)
  - Offline store (training, batch)
  - Point-in-time correctness (no data leakage)

### 10.3 Experiment Tracking
| Tool | What it Tracks |
|------|---------------|
| **MLflow** | Params, metrics, artifacts, models |
| **W&B (Weights & Biases)** | Training curves, hyperparameters |
| **DVC** | Data + model versioning (Git for data) |
| **Neptune** | Team collaboration, metadata |
| **Comet** | Real-time monitoring |

### 10.4 Model Serving
```
Batch Inference:      scheduled, large volumes, offline
Real-time Inference:  low latency, online API
Streaming Inference:  continuous data streams

Serving Frameworks:
- TorchServe (PyTorch)
- TensorFlow Serving
- Triton Inference Server (NVIDIA)
- BentoML
- Ray Serve
- vLLM (LLMs specifically)
- Ollama (local LLMs)
```

### 10.5 CI/CD for ML
```
Code Change (Git push)
  → Lint + Unit Tests
  → Integration Tests
  → Data Validation (Great Expectations)
  → Model Training (triggered)
  → Model Evaluation vs baseline
  → If better: deploy to staging
  → Canary/Blue-Green deployment
  → Full production rollout
```

### 10.6 Model Monitoring
- **Data Drift** — input distribution change ho gayi
- **Concept Drift** — relationship between X and y change ho gaya
- **Performance Degradation** — accuracy drop
- **Data Quality Issues** — nulls, outliers, schema changes
- **Latency Monitoring** — p50, p95, p99 response times

**Tools**: Evidently, Arize AI, Fiddler, Grafana + Prometheus

### 10.7 Infrastructure
```
Local Dev:        Jupyter, VS Code, Docker
Experiment:       GPU instances (A100, H100)
Training:         Kubernetes + Kubeflow / Vertex AI / SageMaker
Serving:          K8s + HPA (autoscaling)
Data Pipeline:    Apache Spark, Kafka, Airflow, Prefect
Storage:          S3, GCS, Azure Blob + Data Lake
```

---

## PART 11: ENTERPRISE AI ARCHITECTURE

### 11.1 LLM Application Stack
```
┌─────────────────────────────────────────┐
│           Frontend / UI                 │
├─────────────────────────────────────────┤
│         API Gateway / Auth              │
├─────────────────────────────────────────┤
│         Orchestration Layer             │
│   (LangChain / LangGraph / Custom)      │
├──────────────┬──────────────────────────┤
│  LLM APIs    │   RAG Pipeline           │
│  OpenAI      │   Vector DB              │
│  Anthropic   │   Embeddings             │
│  Azure OAI   │   Chunking + Indexing    │
├──────────────┴──────────────────────────┤
│         Memory & Cache Layer            │
│  Redis (session) + Semantic Cache       │
├─────────────────────────────────────────┤
│         Data & Document Layer           │
│  S3 / GCS + Document parsers            │
├─────────────────────────────────────────┤
│         Observability                   │
│  Logging + Tracing + Metrics + Alerts   │
└─────────────────────────────────────────┘
```

### 11.2 Multi-Tenant Enterprise Considerations
- **Data Isolation** — har tenant ka data alag
- **Rate Limiting** — per-user/tenant API limits
- **Cost Attribution** — kaunsa tenant kitna spend kar raha hai
- **PII Handling** — GDPR, CCPA compliance
- **Audit Logging** — who asked what, when
- **Access Control** — RBAC (Role-Based Access Control)
- **Encryption** — at rest + in transit

### 11.3 Cost Optimization
- **Prompt Caching** — Anthropic/OpenAI prefix cache (90% cheaper)
- **Semantic Caching** — similar queries → cached answer
- **Model Routing** — simple queries → cheap model; complex → expensive
- **Batching** — requests batch karo (OpenAI Batch API — 50% cheaper)
- **Quantization** — INT4 models run karo (4x cheaper)
- **Self-hosted Models** — Llama, Mistral on own infra

### 11.4 Guardrails & Safety
- **Input Validation** — prompt injection detection
- **Output Validation** — hallucination, toxicity check
- **PII Masking** — before LLM call
- **Content Filtering** — OpenAI Moderation API, Azure Content Safety
- **Rate Limiting** — DDoS protection
- **Fallback Handling** — LLM down ho to kya karo
- **Human-in-the-Loop** — sensitive actions ke liye approval

---

## PART 12: MODERN ARCHITECTURES (2024-2026)

### 12.1 Mixture of Experts (MoE)
- Dense model ki jagah sparse model
- **Gating Network** — input ke liye best "expert" choose karta hai
- Only N experts activate at a time (not all)
- **Models**: GPT-4 (rumored), Mixtral 8x7B, Grok-1
- **Benefit**: More capacity, same compute

### 12.2 State Space Models (SSM)
- Transformers alternative — linear time complexity
- **Mamba** — selective state space, O(n) vs O(n²)
- Long sequences handle karna better
- **Hybrid**: Jamba (Mamba + Transformer)

### 12.3 Multimodal Models
| Model | Modalities |
|-------|-----------|
| GPT-4o | Text + Image + Audio |
| Claude 3 | Text + Image |
| Gemini 1.5 | Text + Image + Video + Audio |
| LLaVA | Text + Image (open source) |
| Whisper | Audio → Text |
| CLIP | Image-Text alignment |

### 12.4 Model Context Protocol (MCP)
- Anthropic ka standard (2024)
- LLM ko tools, resources, prompts connect karna
- Like USB-C for AI tools
- **Components**: MCP Server, MCP Client, Transport
- **Use**: Claude Desktop + GitHub, Slack, Databases

### 12.5 Long Context Strategies
- **RAG** — retrieve relevant parts (most practical)
- **Full Context** — sab kuch window mein daalo (Gemini 1M)
- **Hierarchical Summarization** — chunks → summaries
- **Sliding Window** — local + global attention

---

## PART 13: AI ETHICS & SAFETY

### 13.1 Bias & Fairness
- **Historical Bias** — training data mein already bias tha
- **Representation Bias** — kuch groups underrepresented
- **Measurement Bias** — labels galat hain
- **Mitigation**: Data augmentation, fairness constraints, adversarial debiasing

### 13.2 Explainability (XAI)
| Method | Type | Use |
|--------|------|-----|
| **SHAP** | Model-agnostic | Feature importance globally |
| **LIME** | Local | Single prediction explain karo |
| **Attention Visualization** | DL-specific | Transformer attention maps |
| **GRAD-CAM** | CNN | Image regions important for prediction |
| **Counterfactual** | What-if | "Loan deny hua, kyun?" |

### 13.3 Hallucination
- **Definition**: LLM confident hoke galat information generate karta hai
- **Types**: Factual errors, source fabrication, reasoning errors
- **Mitigation**:
  - RAG (ground in documents)
  - Temperature 0 for factual tasks
  - Citations require karo
  - Verification agents
  - Self-consistency sampling

### 13.4 Privacy Techniques
- **Federated Learning** — data device pe rehta hai, only model updates share
- **Differential Privacy** — noise add karo training mein
- **Homomorphic Encryption** — encrypted data pe compute
- **Secure Multi-party Computation** — parties share kiye bina collaborate

---

## PART 14: TOOLS & ECOSYSTEM

### 14.1 Deep Learning Frameworks
| Framework | Best For |
|-----------|---------|
| **PyTorch** | Research, flexibility, most popular |
| **TensorFlow / Keras** | Production, mobile, Google ecosystem |
| **JAX** | Research, XLA compilation, Google TPUs |
| **ONNX** — framework-agnostic model format

### 14.2 LLM APIs
| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4o, o3 | Most popular |
| **Anthropic** | Claude 3.5/4 | Best for coding, safety |
| **Google** | Gemini 2.0 | Long context, multimodal |
| **Mistral** | Mistral, Mixtral | European, open weights |
| **Meta** | Llama 3.x | Open source, self-host |
| **Groq** | Llama, Mixtral | Ultra-fast inference |
| **Together AI** | Many open models | Serverless |

### 14.3 Orchestration & Integration
- **LangChain** — most popular LLM framework
- **LlamaIndex** — RAG-focused, document ingestion
- **LangGraph** — stateful agent workflows
- **Haystack** — enterprise search + QA
- **Semantic Kernel** — Microsoft, .NET + Python
- **Instructor** — structured outputs from LLMs
- **Guardrails AI** — output validation framework
- **Outlines** — constrained generation

### 14.4 Data Tools
```
Data Collection:    Scrapy, BeautifulSoup, Playwright
Data Processing:    Pandas, Polars, PySpark
Data Validation:    Great Expectations, Pydantic, Pandera
Data Versioning:    DVC, LakeFS, Delta Lake
Annotation:         Label Studio, Scale AI, Prodigy
Pipelines:          Airflow, Prefect, ZenML, Metaflow
```

---

## PART 15: QUICK REFERENCE — WHAT TO USE WHEN

```
Task                           → Solution
─────────────────────────────────────────────────────
Text classification            → BERT fine-tune / GPT few-shot
Named Entity Recognition       → BERT / spaCy
Image classification           → ResNet / EfficientNet / ViT
Object detection               → YOLOv8
Document Q&A                   → RAG + LLM
Chatbot (company knowledge)    → RAG + memory + agents
Code generation                → GPT-4o / Claude / Codestral
Summarization                  → GPT-4o / Claude / Llama
Translation                    → NLLB / DeepL API / GPT
Recommendation                 → Collaborative filtering / DNN
Anomaly detection              → Isolation Forest / Autoencoder
Time series forecasting        → LSTM / Temporal Fusion Transformer
Tabular data (structured)      → XGBoost / LightGBM
Generate images                → Stable Diffusion / DALL-E 3
Speech-to-text                 → Whisper
Text-to-speech                 → ElevenLabs / Azure TTS
Enterprise search              → Elasticsearch + Vector DB
Agentic workflow               → LangGraph + LLM tools
Real-time inference (<50ms)    → vLLM + GPU + KV cache
Local/private LLM              → Ollama + Llama 3 / Mistral
```

---

## PART 16: LEARNING ROADMAP

### Beginner (0-3 months)
1. Python basics (NumPy, Pandas, Matplotlib)
2. Classical ML (scikit-learn)
3. Statistics (probability, distributions, hypothesis testing)
4. Linear algebra, calculus basics

### Intermediate (3-9 months)
1. Deep Learning (PyTorch)
2. CNNs, RNNs, Transformers
3. NLP (Hugging Face, BERT fine-tuning)
4. MLOps basics (MLflow, Docker)
5. Build projects (Kaggle competitions)

### Advanced (9-18 months)
1. LLMs (fine-tuning, prompting, RAG)
2. AI Agents (LangGraph, tools)
3. Production deployment (K8s, vLLM)
4. Research papers (arXiv reading)
5. Distributed training (DeepSpeed, FSDP)

### Enterprise (18+ months)
1. System design for AI at scale
2. Cost optimization strategies
3. Safety & compliance (GDPR, AI Act)
4. Multi-tenant architecture
5. Team leadership & ML culture

---

*Last updated: May 2026 | Covers concepts up to Claude 4, GPT-4o, Gemini 2.0, Llama 3*
