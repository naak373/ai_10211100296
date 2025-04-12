import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import seaborn as sns
import joblib
import tempfile




# Import utility modules
from utils.llm import extract_text_from_pdf, split_text, create_vector_store,rag_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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
    st.divider()
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
    st.divider()

# Neural Network Section
elif page == "Neural Network":

    st.header("Neural Network Training")

    uploaded_file = st.file_uploader("Upload CSV file for neural network training", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select the target variable", df.columns)
        feature_columns = st.multiselect(
            "Select feature variables",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )

        if feature_columns and target_column:
            X = df[feature_columns].copy()
            y = df[target_column].copy()

            is_classification = y.dtype == 'object' or len(np.unique(y)) < 10

            # Handle missing values
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])

            categorical_maps = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le_feat = LabelEncoder()
                    X[col] = le_feat.fit_transform(X[col])
                    categorical_maps[col] = le_feat.classes_

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            le = None
            n_classes = None
            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    y = tf.keras.utils.to_categorical(y)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            st.write("### Model Configuration")
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Epochs", 10, 100, 50)
                batch_size = st.slider("Batch size", 4, 128, 32)
            with col2:
                learning_rate = st.select_slider("Learning rate", options=[0.001, 0.01, 0.1], value=0.01)
                hidden_layers = st.slider("Hidden layers", 1, 5, 2)

            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
            for _ in range(hidden_layers - 1):
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
                          loss=loss, metrics=metrics)

            if st.button("Train Model"):
                st.write("### Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.line_chart()
                metric_chart = st.line_chart()

                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs}")
                        loss_chart.add_rows({'loss': [logs['loss']], 'val_loss': [logs['val_loss']]})
                        if 'accuracy' in logs:
                            metric_chart.add_rows({'accuracy': [logs['accuracy']], 'val_accuracy': [logs['val_accuracy']]})
                        elif 'mae' in logs:
                            metric_chart.add_rows({'mae': [logs['mae']], 'val_mae': [logs['val_mae']]})

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[StreamlitCallback()],
                    verbose=0
                )

                st.write("### Model Evaluation")
                test_results = model.evaluate(X_test, y_test, verbose=0)

                if is_classification:
                    st.metric("Test Accuracy", f"{test_results[1]:.4f}")
                    y_pred = (model.predict(X_test) > 0.5).astype(int) if n_classes == 2 else np.argmax(model.predict(X_test), axis=1)
                    y_true = y_test if n_classes == 2 else np.argmax(y_test, axis=1)

                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.write("### Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose())

                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                else:
                    st.metric("Test MAE", f"{test_results[1]:.4f}")
                    y_pred = model.predict(X_test).flatten()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                # Show final training curves
                st.write("### Final Training Curves")
                history_df = pd.DataFrame(history.history)

                fig_loss, ax1 = plt.subplots()
                ax1.plot(history_df['loss'], label='Train Loss')
                ax1.plot(history_df['val_loss'], label='Val Loss')
                ax1.set_title("Loss Curve")
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Loss")
                ax1.legend()
                st.pyplot(fig_loss)

                if 'accuracy' in history_df.columns:
                    fig_acc, ax2 = plt.subplots()
                    ax2.plot(history_df['accuracy'], label='Train Accuracy')
                    ax2.plot(history_df['val_accuracy'], label='Val Accuracy')
                    ax2.set_title("Accuracy Curve")
                    ax2.set_xlabel("Epochs")
                    ax2.set_ylabel("Accuracy")
                    ax2.legend()
                    st.pyplot(fig_acc)

                elif 'mae' in history_df.columns:
                    fig_mae, ax3 = plt.subplots()
                    ax3.plot(history_df['mae'], label='Train MAE')
                    ax3.plot(history_df['val_mae'], label='Val MAE')
                    ax3.set_title("MAE Curve")
                    ax3.set_xlabel("Epochs")
                    ax3.set_ylabel("MAE")
                    ax3.legend()
                    st.pyplot(fig_mae)

                # Save model + components
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model, \
                     tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_scaler:
                    model.save(tmp_model.name)
                    joblib.dump(scaler, tmp_scaler.name)
                    st.download_button("Download Trained Model (.h5)", open(tmp_model.name, "rb"), "trained_model.h5")
                    st.download_button("Download Scaler", open(tmp_scaler.name, "rb"), "scaler.pkl")
                    if is_classification and le is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_le:
                            joblib.dump(le, tmp_le.name)
                            st.download_button("Download Label Encoder", open(tmp_le.name, "rb"), "label_encoder.pkl")

                # Custom Predictions
                st.write("### Make Custom Predictions")
                input_data = {}
                for col in feature_columns:
                    if col in categorical_maps:
                        user_input = st.selectbox(f"Select value for {col}", categorical_maps[col])
                        input_val = np.where(categorical_maps[col] == user_input)[0][0]
                    else:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        input_val = st.slider(f"Select value for {col}", min_val, max_val, (min_val + max_val) / 2)
                    input_data[col] = input_val

                input_array = np.array([[input_data[col] for col in feature_columns]])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)

                if is_classification:
                    if n_classes == 2:
                        pred_class = "Class 1" if prediction[0] > 0.5 else "Class 0"
                        pred_prob = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
                        st.success(f"Predicted: {pred_class} (Probability: {pred_prob:.4f})")
                    else:
                        pred_idx = np.argmax(prediction)
                        pred_label = le.inverse_transform([pred_idx])[0]
                        st.success(f"Predicted Class: {pred_label} (Confidence: {np.max(prediction):.4f})")
                else:
                    st.success(f"Predicted Value: {prediction[0][0]:.4f}")

        st.divider()



# LLM section
elif page == "LLM Solution":
    st.header("Large Language Model Q&A")

    # Architecture explanation with custom diagram
    with st.expander("LLM Architecture"):
        st.markdown("""
        ### RAG (Retrieval-Augmented Generation) Architecture for Student Handbook

        This solution uses a custom RAG pipeline to answer questions from the Acity Student Handbook:

        1. **Document Loading**: Extract raw text from the handbook PDF using PyPDF.
        2. **Text Chunking**: Split the text into overlapping chunks (~500 words).
        3. **Embedding Generation**: Generate vector embeddings using a HuggingFace model.
        4. **Vector Indexing**: Store chunk embeddings in a FAISS vector index.
        5. **Query Embedding**: Convert the user's question into an embedding.
        6. **Similarity Search**: Retrieve top‑k relevant chunks based on vector similarity.
        7. **Prompt Construction**: Assemble context + query into a prompt.
        8. **Answer Generation**: Use Mistral-7B-Instruct to generate a response based on retrieved content.
        """)
        st.image("rag_architecture.png", caption="RAG Architecture for Student Handbook Q&A")

    # Methodology explanation
    with st.expander("Methodology"):
        st.markdown("""
        ### Methodology: RAG Pipeline with Acity Student Handbook

        #### 1. Text Extraction
        - Extracts text using `extract_text_from_pdf()` via PyPDF.

        #### 2. Text Chunking
        - Splits text into ~500-word chunks using a custom `split_text()` function with overlap.

        #### 3. Embedding & Indexing
        - Uses a HuggingFace embedding model (e.g., `all-MiniLM-L6-v2`) to generate vector representations of each chunk.
        - Stores embeddings in a FAISS index using `create_vector_store()`.

        #### 4. Retrieval & Generation
        - Embeds user query and finds similar chunks using FAISS.
        - `rag_pipeline()` assembles a prompt from top-k chunks and sends it to the Mistral-7B-Instruct API.
        """)

    # Get file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "data", "handbook.pdf")

    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at path: {pdf_path}")
        st.write("Current working directory: " + os.getcwd())
        st.stop()

    # Cache vector store in session
    if 'vector_store' not in st.session_state:
        with st.spinner("Loading student handbook... This may take a minute"):
            handbook_text = extract_text_from_pdf(pdf_path)
            chunks = split_text(handbook_text)
            st.session_state.vector_store = create_vector_store(chunks)

    # User query input
    query = st.text_input("Ask a question about student policies:")

    if query:
        with st.spinner("Searching handbook and generating answer..."):
            response_mistral = rag_pipeline(query, st.session_state.vector_store)

            # Clean up response formatting
            if "[/INST]" in response_mistral:
                response_mistral = response_mistral.split("[/INST]")[1].strip()
                if response_mistral.startswith("</s>"):
                    response_mistral = response_mistral.replace("</s>", "", 1).strip()

        st.write("### Query Response")
        st.success(response_mistral)

        st.divider()




           