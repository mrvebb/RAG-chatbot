import os
import streamlit as st
from typing import List

# GEREKLİ TÜM IMPORT'LARIN DOĞRU YERLERDEN YAPILMASI:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint # StopIteration hatasını çözmek için 'Hub' yerine 'Endpoint' kullanacağız
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Embedding Modelini Yükle (Aynı, bu hala lokal ve hızlı)
@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Vektör Veritabanını Oluştur (Aynı, cache düzeltmesiyle)
@st.cache_resource(show_spinner="Veri seti işleniyor...")
def build_vectorstore(text: str, _embeddings: HuggingFaceEmbeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks: List[str] = splitter.split_text(text)
    return FAISS.from_texts(chunks, embedding=_embeddings)

# 3. RAG Zincirini Oluştur (Hugging Face API ile - DÜZELTİLMİŞ)
@st.cache_resource(show_spinner="Yapay zeka modeli (API) yükleniyor...")
# @st.cache_resource(show_spinner="Yapay zeka modeli (API) yükleniyor...")
def build_rag_chain(_vectorstore: FAISS, api_token: str):
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # === HATANIN ÇÖZÜMÜ BURADA ===
    # 'HuggingFaceHub' yerine modern 'HuggingFaceEndpoint' sınıfını kullanıyoruz
    # ve API anahtarını ('api_token') doğrudan 'huggingfacehub_api_token' parametresi ile iletiyoruz.
    
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=api_token,
    temperature=0.1,
    max_new_tokens=250,
    task="text-generation"
)




    # === HATA ÇÖZÜLDÜ ===

    # Mistral bir "instruct" modeli olduğu için Chat-bazlı prompt'u anlar.
    system_prompt = (
        "Aşağıda verilen bağlam parçalarını kullanarak, kullanıcının sorusuna"
        " doğru ve kısa bir yanıt ver. Bilmediğinde uydurma. Gerekirse 'Elimdeki bağlamda"
        " bu bilgi yok' de.\n\nBağlam:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Soru: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain

def main():
    st.set_page_config(page_title="Basit RAG (Hugging Face API)")
    st.title("Basit RAG Uygulaması (Hugging Face API)")
    st.markdown("Bu uygulama, ücretsiz Hugging Face API'sini kullanmaktadır.")

    # API anahtarı st.secrets üzerinden alınır (HUGGINGFACE_API_TOKEN)
    if "HUGGINGFACE_API_TOKEN" not in st.secrets or not st.secrets["HUGGINGFACE_API_TOKEN"]:
        st.warning("Lütfen .streamlit/secrets.toml içine HUGGINGFACE_API_TOKEN ekleyin.")
        return
    api_key = st.secrets["HUGGINGFACE_API_TOKEN"]

    # Metni yükle
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.error("data.txt bulunamadı. Lütfen proje klasörüne ekleyin.")
        return

    # Embedding ve vektör veritabanını hazırla
    embeddings = load_embeddings()
    vectorstore = build_vectorstore(raw_text, embeddings)

    # RAG zincirini oluştur (anahtarı vererek)
    rag_chain = build_rag_chain(vectorstore, api_key)

    # Kullanıcı girişi
    question = st.text_input("Sorunuzu yazın")
    if question:
        with st.spinner("Yanıt üretiliyor..."):
            result = rag_chain.invoke(question)
            st.write(result)

if __name__ == "__main__":
    main()