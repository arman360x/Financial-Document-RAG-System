"""
Streamlit Web Interface for SEC RAG System

Provides an interactive dashboard for:
- Querying SEC filings
- Viewing source citations
- Analytics and metrics
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

import config
from rag_chain import RAGChain, get_rag_chain
from retriever import get_retriever
from vector_store import get_vector_store

# Page configuration
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout=config.STREAMLIT_LAYOUT,
)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


@st.cache_resource
def initialize_system():
    """Initialize RAG system components."""
    vector_store = get_vector_store()
    retriever = get_retriever(vector_store=vector_store)
    rag_chain = get_rag_chain(retriever=retriever)
    return vector_store, rag_chain


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📊 SEC Filings RAG Intelligence")
    st.markdown(
        "Ask questions about company financials, risk factors, "
        "and more based on SEC filings."
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Filter options
        st.subheader("Filters")
        
        ticker_filter = st.selectbox(
            "Company",
            options=["All"] + list(config.TICKER_TO_CIK.keys()),
            index=0,
        )
        
        filing_type_filter = st.selectbox(
            "Filing Type",
            options=["All", "10-K", "10-Q", "8-K"],
            index=0,
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_hybrid = st.checkbox("Use Hybrid Search", value=False)
            use_reranking = st.checkbox("Enable Reranking", value=True)
            max_sources = st.slider("Max Sources", 1, 10, 5)
        
        st.divider()
        
        # System status
        st.subheader("📈 System Status")
        
        try:
            vector_store, rag_chain = initialize_system()
            stats = vector_store.get_stats()
            
            st.metric("Total Documents", f"{stats['total_chunks']:,}")
            st.metric("Companies Indexed", stats['unique_tickers'])
            
            st.success("✅ System Ready")
            
        except Exception as e:
            st.error(f"❌ System Error: {str(e)}")
            st.info("Please ensure the database is running and populated.")
            return
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "📚 Sources"])
    
    with tab1:
        render_chat_interface(rag_chain, ticker_filter, filing_type_filter, max_sources)
    
    with tab2:
        render_analytics_dashboard(vector_store)
    
    with tab3:
        render_source_explorer(vector_store)


def render_chat_interface(rag_chain, ticker_filter, filing_type_filter, max_sources):
    """Render the chat interface."""
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['source']}** "
                            f"(relevance: {source['relevance_score']:.2f})"
                        )
                        if source.get("url"):
                            st.markdown(f"  [View Filing]({source['url']})")
    
    # Chat input
    if prompt := st.chat_input("Ask about SEC filings..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Build filters
        filters = {}
        if ticker_filter != "All":
            filters["ticker"] = ticker_filter
        if filing_type_filter != "All":
            filters["filing_type"] = filing_type_filter
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching SEC filings..."):
                try:
                    response = rag_chain.query(
                        prompt,
                        filters=filters if filters else None,
                    )
                    
                    st.markdown(response.answer)
                    
                    # Show confidence
                    confidence_color = (
                        "green" if response.confidence > 0.7
                        else "orange" if response.confidence > 0.4
                        else "red"
                    )
                    st.markdown(
                        f"<small>Confidence: "
                        f"<span style='color:{confidence_color}'>"
                        f"{response.confidence:.0%}</span></small>",
                        unsafe_allow_html=True,
                    )
                    
                    # Show sources
                    if response.sources:
                        with st.expander("📚 View Sources"):
                            for source in response.sources:
                                st.markdown(
                                    f"- **{source['source']}** "
                                    f"(relevance: {source['relevance_score']:.2f})"
                                )
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "confidence": response.confidence,
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Example queries
    st.divider()
    st.markdown("**Example queries:**")
    
    example_queries = [
        "What are Apple's main risk factors?",
        "How much revenue did Microsoft report last year?",
        "What legal proceedings is Tesla involved in?",
        "Summarize Amazon's business strategy",
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(query, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()


def render_analytics_dashboard(vector_store):
    """Render analytics dashboard."""
    
    st.header("📊 Analytics Dashboard")
    
    try:
        stats = vector_store.get_stats()
    except Exception:
        st.warning("Could not load analytics. Please ensure database is connected.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", f"{stats['total_chunks']:,}")
    
    with col2:
        st.metric("Companies", stats['unique_tickers'])
    
    with col3:
        st.metric("Filing Types", len(stats['filing_types']))
    
    with col4:
        st.metric("Index Status", "Ready ✅")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Documents by Company")
        
        # This would need actual data from the database
        # For demo, showing placeholder data
        company_data = {
            "AAPL": 245,
            "MSFT": 312,
            "GOOGL": 198,
            "AMZN": 276,
            "TSLA": 167,
        }
        
        fig = px.bar(
            x=list(company_data.keys()),
            y=list(company_data.values()),
            labels={"x": "Company", "y": "Document Chunks"},
            color=list(company_data.values()),
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Documents by Filing Type")
        
        filing_data = {
            "10-K": 450,
            "10-Q": 380,
            "8-K": 170,
        }
        
        fig = px.pie(
            values=list(filing_data.values()),
            names=list(filing_data.keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Query performance
    st.subheader("Query Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recent Queries**")
        if st.session_state.messages:
            user_queries = [
                m["content"] for m in st.session_state.messages
                if m["role"] == "user"
            ][-5:]
            for q in user_queries:
                st.markdown(f"- {q}")
        else:
            st.info("No queries yet")
    
    with col2:
        st.markdown("**Confidence Distribution**")
        confidences = [
            m.get("confidence", 0) for m in st.session_state.messages
            if m["role"] == "assistant" and "confidence" in m
        ]
        if confidences:
            fig = px.histogram(
                x=confidences,
                nbins=10,
                labels={"x": "Confidence", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet")


def render_source_explorer(vector_store):
    """Render source document explorer."""
    
    st.header("📚 Source Explorer")
    
    st.markdown(
        "Browse indexed SEC filings and their sections."
    )
    
    try:
        stats = vector_store.get_stats()
        
        # Company selector
        selected_company = st.selectbox(
            "Select Company",
            options=stats.get('tickers', []),
        )
        
        if selected_company:
            st.subheader(f"Filings for {selected_company}")
            
            # This would show actual filing data
            st.info(
                f"Showing indexed documents for {selected_company}. "
                f"Use the search functionality to query specific content."
            )
            
            # Filing type breakdown
            st.markdown("**Available Filing Types:**")
            for filing_type in stats.get('filing_types', []):
                st.markdown(f"- {filing_type}")
        
    except Exception as e:
        st.warning(f"Could not load sources: {str(e)}")
    
    st.divider()
    
    # Direct search
    st.subheader("🔍 Direct Search")
    
    search_query = st.text_input("Search document content:")
    
    if search_query and st.button("Search"):
        st.info("Search functionality requires retriever initialization.")


# Run the app
if __name__ == "__main__":
    main()
