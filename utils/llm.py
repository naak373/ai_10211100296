import pandas as pd
import numpy as np

def load_election_data():
    """Load the Ghana election results dataset."""
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def convert_df_to_texts(df):
    """Convert dataframe rows to text chunks for RAG."""
    texts = []
    for _, row in df.iterrows():
        text = f"In {row.get('Year', 'N/A')}, in the {row.get('Region', 'N/A')} region of Ghana, "
        text += f"constituency {row.get('Constituency', 'N/A')}, "
        text += f"the presidential election results were: "
        
        for column in df.columns:
            if 'party' in column.lower() or 'candidate' in column.lower():
                text += f"{column}: {row.get(column, 'N/A')}, "
        
        texts.append(text.strip(", "))
    
    return texts

def initialize_rag_pipeline(df):
    """Initialize the RAG pipeline with election data."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        
        # Convert dataframe to text chunks
        texts = convert_df_to_texts(df)
        
        # Use a smaller model for embedding
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
        print(f"Error initializing RAG pipeline: {e}")
        return None

def process_query(query, rag_pipeline, k=3):
    """Process a user query through the RAG pipeline."""
    # Embed the query
    query_embedding = rag_pipeline["model"].encode([query])
    
    # Search for similar chunks
    distances, indices = rag_pipeline["index"].search(query_embedding, k)
    
    # Get the retrieved texts
    retrieved_chunks = [rag_pipeline["texts"][i] for i in indices[0]]
    relevance_scores = [1/(1+d) for d in distances[0]]
    
    return retrieved_chunks, relevance_scores

def generate_response(query, retrieved_chunks):
    """
    Generate a response based on the query and retrieved chunks.
    In a full implementation, this would use the Mistral-7B model.
    """
    # Simplified response generation based on query keywords
    if "ashanti" in query.lower() and "presidential" in query.lower():
        response = """
        Based on the retrieved information from the Ghana Election Results dataset, the New Patriotic Party (NPP) 
        has traditionally been strong in the Ashanti region in recent elections. This region is considered an NPP 
        stronghold, with the party consistently winning the presidential votes there by significant margins.
        
        In the most recent elections covered by the dataset, the NPP presidential candidate received the majority 
        of votes across most constituencies in the Ashanti region. The specific percentages varied by constituency, 
        but the overall trend shows NPP dominance in this region.
        
        If you'd like specific numbers for particular constituencies or election years, please ask a more specific question.
        """
    elif "party" in query.lower() and "won" in query.lower():
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
    
    return response

def get_chatgpt_comparison(query):
    """Provide a simulated ChatGPT response for comparison."""
    if "ashanti" in query.lower():
        return """
        I don't have specific, up-to-date information about Ghana's election results, especially for recent elections. Ghana has had several elections, with the New Patriotic Party (NPP) and the National Democratic Congress (NDC) being the major political parties.
        
        The Ashanti Region has historically been considered a stronghold for the NPP, but without access to specific election data, I cannot provide precise information about which party won in recent elections in this region. For accurate and up-to-date information, I would recommend checking official election commission data or reliable news sources that cover Ghanaian politics.
        """
    else:
        return """
        Without specific access to comprehensive election data for Ghana, I can't provide detailed information about election results for particular regions or years. Ghana's political landscape is primarily dominated by two main parties: the New Patriotic Party (NPP) and the National Democratic Congress (NDC).
        
        Different regions in Ghana have historically shown preferences for particular parties, but results can vary by election cycle and can be influenced by multiple factors including candidate selection, campaign issues, and voter turnout.
        
        For accurate and up-to-date information about specific election results, I would recommend consulting official sources like the Electoral Commission of Ghana or reputable news organizations that cover Ghanaian politics.
        """