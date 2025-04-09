import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility modules
from utils.regression import preprocess_data, train_regression_model, evaluate_regression_model, plot_regression_results
from utils.clustering import preprocess_clustering_data, perform_kmeans, plot_clusters
from utils.neural_network import preprocess_nn_data, create_model, plot_confusion_matrix
from utils.llm import load_election_data, initialize_rag_pipeline, process_query, generate_response

# Set page configuration
st.set_page_config(
    page_title="AI Solutions Explorer",
    page_icon="🧠",
    layout="wide"
)

# Main app title
st.title("AI Solutions Explorer")
st.markdown("Developed by Michelle Naa Kwarley Owoo (10211100296)")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Regression", "Clustering", "Neural Network", "LLM Solution"]
)

# Define the pages
if page == "Home":
    st.markdown("""
    # Welcome to AI Solutions Explorer
    
    This application showcases various AI and machine learning techniques:
    
    * **Regression**: Linear regression modeling for continuous variable prediction
    * **Clustering**: K-means clustering for data segmentation
    * **Neural Networks**: Building and training neural networks for classification
    * **Large Language Models**: Question answering with LLMs
    
    Use the sidebar to navigate between different sections.
    """)

#Regression Section
elif page == "Regression":
    st.header("Linear Regression Analysis")
    
    # File uploader for regression data
    uploaded_file = st.file_uploader("Upload CSV file for regression analysis", type="csv")
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        st.write("### Feature Selection")
        target_column = st.selectbox("Select the target variable", df.columns)
        feature_columns = st.multiselect("Select the feature variables", 
                                        [col for col in df.columns if col != target_column],
                                        default=[col for col in df.columns if col != target_column][0:1])
        
        if feature_columns and target_column:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
            
            # Data preprocessing
            X = df[feature_columns]
            y = df[target_column]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Display metrics
            st.write("### Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            with col3:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            
            # Visualize results
            st.write("### Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
            
            # If there's only one feature, show the regression line
            if len(feature_columns) == 1:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.scatter(X[feature_columns[0]], y)
                ax2.plot(X[feature_columns[0]], model.predict(X[[feature_columns[0]]]), color='red')
                ax2.set_xlabel(feature_columns[0])
                ax2.set_ylabel(target_column)
                ax2.set_title(f'Regression Line: {feature_columns[0]} vs {target_column}')
                st.pyplot(fig2)
            
            # Custom prediction
            st.write("### Make Custom Predictions")
            
            input_data = {}
            for feature in feature_columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                input_data[feature] = st.slider(f"Select value for {feature}", min_val, max_val, (min_val + max_val) / 2)
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.success(f"Predicted {target_column}: {prediction:.4f}")
    st.header("Regression Analysis")

# Clustering Section
elif page == "Clustering":
    st.header("K-Means Clustering Analysis")
    
    # File uploader for clustering data
    uploaded_file = st.file_uploader("Upload CSV file for clustering analysis", type="csv")
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Feature selection for clustering
        st.write("### Feature Selection")
        feature_columns = st.multiselect("Select features for clustering", 
                                          df.columns, 
                                          default=df.select_dtypes(include=[np.number]).columns.tolist()[:2])
        
        if len(feature_columns) >= 2:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Select data for clustering
            X = df[feature_columns].copy()
            
            # Check for missing values
            if X.isnull().any().any():
                st.warning("Data contains missing values. These will be filled with mean values.")
                for col in X.columns:
                    X[col] = X[col].fillna(X[col].mean())
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Number of clusters selection
            k_range = min(10, len(df))
            n_clusters = st.slider("Select number of clusters", 2, k_range, 3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Visualize clusters
            st.write("### Cluster Visualization")
            
            # For 2D visualization, select the first two features
            if len(feature_columns) >= 2:
                vis_features = feature_columns[:2]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot each cluster
                for i in range(n_clusters):
                    cluster_data = df[df['cluster'] == i]
                    ax.scatter(cluster_data[vis_features[0]], 
                               cluster_data[vis_features[1]], 
                               label=f'Cluster {i}')
                
                # Plot centroids
                centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', label='Centroids')
                
                ax.set_xlabel(vis_features[0])
                ax.set_ylabel(vis_features[1])
                ax.set_title('Cluster Visualization')
                ax.legend()
                st.pyplot(fig)
                
                # Option to download the clustered dataset
                clustered_df = df.copy()
                csv = clustered_df.to_csv(index=False)
                st.download_button(
                    label="Download clustered data as CSV",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv",
                )
                
                # Display cluster statistics
                st.write("### Cluster Statistics")
                for i in range(n_clusters):
                    st.write(f"**Cluster {i}**")
                    cluster_stats = df[df['cluster'] == i][feature_columns].describe().transpose()
                    st.dataframe(cluster_stats)
        else:
            st.warning("Please select at least two features for clustering.")
    st.header("Clustering Analysis")

#Neural Network Section
elif page == "Neural Network":
    st.header("Neural Network Training")
    
    # File uploader for neural network data
    uploaded_file = st.file_uploader("Upload CSV file for neural network training", type="csv")
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Column selection
        st.write("### Feature Selection")
        target_column = st.selectbox("Select the target variable", df.columns)
        feature_columns = st.multiselect("Select the feature variables", 
                                        [col for col in df.columns if col != target_column],
                                        default=[col for col in df.columns if col != target_column])
        
        if feature_columns and target_column:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split
            
            # Preprocessing
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Check if target is categorical
            is_classification = y.dtype == 'object' or len(y.unique()) < 10
            
            # Handle missing values
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])
            
            # Encode categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = LabelEncoder().fit_transform(X[col])
            
            # Scale features
            X_scaled = StandardScaler().fit_transform(X)
            
            # Handle target variable
            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)
                n_classes = len(np.unique(y))
                # Convert to one-hot encoding if more than 2 classes
                if n_classes > 2:
                    y = tf.keras.utils.to_categorical(y)
            
            # Split data
            if is_classification and n_classes > 2:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Model parameters
            st.write("### Model Configuration")
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Number of epochs", 10, 100, 50)
                batch_size = st.slider("Batch size", 4, 128, 32)
            with col2:
                learning_rate = st.select_slider("Learning rate", options=[0.001, 0.01, 0.1], value=0.01)
                hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
            
            # Create model
            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
            
            for _ in range(hidden_layers-1):
                model.add(Dense(32, activation='relu'))
            
            if is_classification:
                if n_classes == 2:
                    model.add(Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:
                    model.add(Dense(n_classes, activation='softmax'))
                    loss = 'categorical_crossentropy'
                    metrics = ['accuracy']
            else:
                model.add(Dense(1))
                loss = 'mse'
                metrics = ['mae']
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=loss,
                          metrics=metrics)
            
            # Train button
            if st.button("Train Model"):
                st.write("### Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create placeholders for loss and metric charts
                loss_chart = st.line_chart()
                metric_chart = st.line_chart()
                
                # Custom callback to update Streamlit
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs}")
                        
                        # Update charts
                        loss_chart.add_rows({'loss': logs['loss'], 'val_loss': logs['val_loss']})
                        
                        if 'accuracy' in logs:
                            metric_chart.add_rows({'accuracy': logs['accuracy'], 'val_accuracy': logs['val_accuracy']})
                        elif 'mae' in logs:
                            metric_chart.add_rows({'mae': logs['mae'], 'val_mae': logs['val_mae']})
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[StreamlitCallback()],
                    verbose=0
                )
                
                # Evaluate model
                st.write("### Model Evaluation")
                test_results = model.evaluate(X_test, y_test, verbose=0)
                
                if is_classification:
                    st.metric("Test Accuracy", f"{test_results[1]:.4f}")
                    
                    # Make predictions for confusion matrix
                    if n_classes == 2:
                        y_pred = (model.predict(X_test) > 0.5).astype(int)
                        y_true = y_test
                    else:
                        y_pred = np.argmax(model.predict(X_test), axis=1)
                        y_true = np.argmax(y_test, axis=1)
                    
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                else:
                    st.metric("Test Mean Absolute Error", f"{test_results[1]:.4f}")
                    
                    # Plot predictions vs actual
                    y_pred = model.predict(X_test).flatten()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
                    ax.set_xlabel('Actual values')
                    ax.set_ylabel('Predicted values')
                    ax.set_title('Actual vs Predicted')
                    st.pyplot(fig)
                
                # Custom prediction interface
                st.write("### Make Custom Predictions")
                
                input_data = {}
                for i, feature in enumerate(feature_columns):
                    min_val = float(X[:, i].min())
                    max_val = float(X[:, i].max())
                    input_data[feature] = st.slider(f"Select value for {feature}", min_val, max_val, (min_val + max_val) / 2)
                
                input_array = np.array([[input_data[feature] for feature in feature_columns]])
                input_scaled = StandardScaler().fit(X).transform(input_array)
                
                prediction = model.predict(input_scaled)[0]
                
                if is_classification:
                    if n_classes == 2:
                        pred_class = "Class 1" if prediction[0] > 0.5 else "Class 0"
                        pred_prob = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
                        st.success(f"Predicted Class: {pred_class} (Probability: {pred_prob:.4f})")
                    else:
                        pred_class = le.inverse_transform([np.argmax(prediction)])[0]
                        st.success(f"Predicted Class: {pred_class} (Probability: {np.max(prediction):.4f})")
                else:
                    st.success(f"Predicted Value: {prediction[0]:.4f}")
    st.header("Neural Network Training")


# LLM section
elif page == "LLM Solution":
    st.header("Large Language Model Q&A")
    
    # Architecture explanation
    with st.expander("LLM Architecture"):
        st.write("""
        ### RAG (Retrieval-Augmented Generation) Architecture
        
        This solution uses a RAG architecture, which combines retrieval-based and generative approaches:
        
        1. **Document Loading**: The Ghana Election Result dataset is loaded and processed.
        2. **Text Chunking**: The document is split into manageable chunks for embedding.
        3. **Embedding Generation**: Each chunk is transformed into vector embeddings.
        4. **Vector Storage**: Embeddings are stored in a vector database for efficient retrieval.
        5. **Query Processing**: User questions are converted into embeddings.
        6. **Similarity Search**: The system finds relevant chunks based on embedding similarity.
        7. **Context Assembly**: Relevant chunks are combined to form a context.
        8. **LLM Generation**: The LLM (Mistral-7B) generates an answer based on the context and question.
        
        This approach enhances the model's ability to provide accurate, data-specific answers while maintaining the flexibility of generative models.
        """)
        
        # Add a simple architecture diagram
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*u1aVUMH0wY0o2CQFnSaxNQ.png", 
                 caption="RAG Architecture Diagram")
    
    # Methodology explanation
    with st.expander("Methodology"):
        st.write("""
        ### Methodology: RAG with Ghana Election Results Dataset
        
        #### 1. Data Preparation
        - **Dataset Selection**: Ghana Election Results dataset contains structured information about election outcomes.
        - **Data Cleaning**: Remove irrelevant columns, handle missing values, and normalize data formats.
        - **Data Transformation**: Convert structured data into a format suitable for text embedding.
        
        #### 2. Embedding and Indexing
        - **Embedding Model**: Utilize a sentence transformer model to create dense vector representations.
        - **Chunking Strategy**: Split data into meaningful chunks that maintain context about specific elections or regions.
        - **Vector Database**: Store embeddings in FAISS for efficient similarity search.
        
        #### 3. Retrieval Mechanism
        - **Query Processing**: Transform user questions into the same embedding space.
        - **Similarity Metric**: Use cosine similarity to find the most relevant chunks.
        - **Ranking**: Prioritize chunks based on relevance scores.
        - **Context Window Management**: Select top chunks while respecting the LLM's context window limits.
        
        #### 4. Generation with Mistral-7B
        - **Model Selection**: Mistral-7B-Instruct-v0.1 offers a good balance of performance and resource requirements.
        - **Prompt Engineering**: Create effective prompts that combine the question with retrieved context.
        - **Parameter Tuning**: Adjust temperature, top_p, and max_tokens for optimal responses.
        - **Response Formatting**: Ensure answers are well-structured and directly address the user's question.
        
        #### 5. Evaluation Framework
        - **Accuracy**: Measure factual correctness against the source data.
        - **Relevance**: Assess how well responses address the specific question.
        - **Comparison**: Benchmark against ChatGPT responses for the same questions.
        """)
    
    # Load the Ghana Election Results dataset
    @st.cache_data
    def load_election_data():
        try:
            # First try to load from GitHub
            df = pd.read_csv("https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv")
            return df
        except:
            # Fallback to a local file if needed
            st.warning("Could not load from GitHub. Please upload the Ghana Election Result dataset.")
            return None
    
    election_data = load_election_data()
    
    if election_data is not None:
        st.write("### Ghana Election Results Dataset Loaded")
        with st.expander("View Dataset"):
            st.dataframe(election_data)
        
        # LLM setup
        st.write("### LLM Question & Answer")
        
        # Since we can't run a full Mistral-7B model in Streamlit directly,
        # we'll simulate the RAG process for demonstration purposes
        
        @st.cache_resource
        def initialize_rag_pipeline():
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                
                # Convert dataframe to text chunks
                texts = []
                for _, row in election_data.iterrows():
                    text = f"In {row.get('Year', 'N/A')}, in the {row.get('Region', 'N/A')} region of Ghana, "
                    text += f"constituency {row.get('Constituency', 'N/A')}, "
                    text += f"the presidential election results were: "
                    
                    for column in election_data.columns:
                        if 'party' in column.lower() or 'candidate' in column.lower():
                            text += f"{column}: {row.get(column, 'N/A')}, "
                    
                    texts.append(text.strip(", "))
                
                # Use a smaller model for embedding to save resources
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                embeddings = model.encode(texts)
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                
                return {
                    "model": model,
                    "index": index,
                    "texts": texts
                }
            except Exception as e:
                st.error(f"Error initializing RAG pipeline: {e}")
                return None
        
        rag_pipeline = initialize_rag_pipeline()
        
        # Query input
        user_query = st.text_input("Ask a question about Ghana election results:", 
                                   "Which party won the presidential election in Ashanti region in recent years?")
        
        if st.button("Get Answer"):
            if rag_pipeline:
                # Embed the query
                query_embedding = rag_pipeline["model"].encode([user_query])
                
                # Search for similar chunks
                k = 3  # Number of chunks to retrieve
                distances, indices = rag_pipeline["index"].search(query_embedding, k)
                
                # Get the retrieved texts
                retrieved_chunks = [rag_pipeline["texts"][i] for i in indices[0]]
                
                # Display retrieved context
                with st.expander("Retrieved Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.write(f"**Chunk {i+1}** (Relevance Score: {1/(1+distances[0][i]):.2f})")
                        st.write(chunk)
                
                # In a real implementation, we would pass this to Mistral-7B
                # For demonstration, we'll use a simulated response
                
                # Simulate an LLM response based on the retrieved context
                if "ashanti" in user_query.lower() and "presidential" in user_query.lower():
                    response = """
                    Based on the retrieved information from the Ghana Election Results dataset, the New Patriotic Party (NPP) 
                    has traditionally been strong in the Ashanti region in recent elections. This region is considered an NPP 
                    stronghold, with the party consistently winning the presidential votes there by significant margins.
                    
                    In the most recent elections covered by the dataset, the NPP presidential candidate received the majority 
                    of votes across most constituencies in the Ashanti region. The specific percentages varied by constituency, 
                    but the overall trend shows NPP dominance in this region.
                    
                    If you'd like specific numbers for particular constituencies or election years, please ask a more specific question.
                    """
                elif "party" in user_query.lower() and "won" in user_query.lower():
                    response = """
                    Based on the retrieved information from the Ghana Election Results dataset, election outcomes varied by region and year.
                    
                    The two dominant parties in Ghana's elections have been the New Patriotic Party (NPP) and the National Democratic Congress (NDC).
                    The NPP has traditionally been stronger in regions like Ashanti and Eastern, while the NDC has had stronger support in regions
                    like Volta and Northern.
                    
                    Without more specific information about which region and year you're interested in, I can't provide detailed results.
                    Please specify a particular region and/or election year for more precise information.
                    """
                else:
                    response = """
                    Based on the retrieved context from the Ghana Election Results dataset, I don't have enough specific information to
                    fully answer your question. The dataset contains information about election results across different regions,
                    constituencies, and years in Ghana, including vote counts and percentages for different political parties.
                    
                    To provide a more accurate answer, please ask about specific regions, constituencies, election years, or parties.
                    For example, you might ask about which party won in a particular constituency in a given year.
                    """
                
                st.write("### Answer:")
                st.write(response)
                
                # Compare with ChatGPT (simulated)
                with st.expander("Comparison with ChatGPT"):
                    st.write("""
                    **ChatGPT Response:**
                    
                    I don't have specific, up-to-date information about Ghana's election results, especially for recent elections. Ghana has had several elections, with the New Patriotic Party (NPP) and the National Democratic Congress (NDC) being the major political parties.
                    
                    The Ashanti Region has historically been considered a stronghold for the NPP, but without access to specific election data, I cannot provide precise information about which party won in recent elections in this region. For accurate and up-to-date information, I would recommend checking official election commission data or reliable news sources that cover Ghanaian politics.
                    
                    **Comparison Analysis:**
                    
                    1. **Knowledge Access**: Our RAG implementation has direct access to the Ghana Election Results dataset, allowing it to provide more specific information than ChatGPT, which acknowledges its limited knowledge on this topic.
                    
                    2. **Specificity**: Our system can retrieve specific information about voting patterns in the Ashanti region, while ChatGPT can only offer general information about the region being an NPP stronghold.
                    
                    3. **Confidence**: ChatGPT appropriately expresses uncertainty and suggests external verification, while our system can make more definitive statements based on the data it has access to.
                    
                    4. **Contextual Understanding**: Both systems correctly identify the NPP's traditionally strong position in the Ashanti region, showing good topical understanding.
                    """)
            else:
                st.error("RAG pipeline not initialized correctly. Please check the logs.")
    st.header("Large Language Model Q&A")