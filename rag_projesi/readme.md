1) Projenin Amacı

Kullanıcının doğal dil sorularına, önceden sağlanmış metin bağlamından benzer parçaları (retrieval) bularak ve bu bağlamı kullanarak bir LLM ile cevap üretip (generation) Streamlit üzerinden sunmak. Amaç: küçük ve çalıştırması kolay bir RAG örneği sağlamak; eğitim/demo amaçlı kullanılabilir.

2) Veri Seti Hakkında

data.txt dosyası düz metin (UTF-8) formatındadır.

İçinde proje konusuna dair dökümantasyon, SSS, makale özetleri, kullanım kılavuzları vb. olabilir.

İdeal olarak uzun belgeleri parçalara bölmek için satır/başlık ayırıcıları kullanın (ör. \n\n ile paragraflar).

Örnek (data.txt):

Proje tanıtımı: Bu belge RAG tabanlı chatbot örneğidir...
SSS:
Soru: Uygulamayı nasıl çalıştırırım?
Cevap: ...


3) Kullanılan Yöntemler / Teknoloji Stack

Arayüz: Streamlit

Metin bölme: RecursiveCharacterTextSplitter (langchain_text_splitters)

Embedding: sentence-transformers/all-MiniLM-L6-v2 (HuggingFaceEmbeddings)

Vektör DB: FAISS (faiss-cpu)

Model (generation): HuggingFace endpoint (ör. HuggingFaceH4/zephyr-7b-beta) — Hugging Face API token gerektirir

RAG akışı: Retriever (k-NN benzerlik) → Bağlam birleştirme → Prompt → LLM
4) Gereksinimler / Kurulum

Python 3.9+ (tercihen 3.10)

Sanal ortam oluşturup aktifleştirin:

Windows:

python -m venv .venv
.venv\Scripts\activate


macOS / Linux:

python -m venv .venv
source .venv/bin/activate


Paketleri kurun:

pip install -r requirements.txt


Örnek requirements.txt:

streamlit
langchain
langchain-core
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
huggingface-hub


Not: GPU kullanıyorsanız faiss-gpu tercih edebilirsiniz.

5) Hugging Face API Anahtarı (Gizli)

Hugging Face hesabınızdan bir access token oluşturun.

Proje root içinde .streamlit/secrets.toml dosyası oluşturun ve içine:

HUGGINGFACE_API_TOKEN = "hf_...."


.gitignore dosyanızda .streamlit/secrets.toml'i ekleyin — token'ı asla public repoya koymayın.

6) Uygulamayı Çalıştırma (Adım Adım)

Sanal ortam aktif, paketler kurulu ve data/data.txt var olduğundan emin olun.

.streamlit/secrets.toml içine HUGGINGFACE_API_TOKEN ekleyin.

Uygulamayı başlatın:

streamlit run app/main.py


Tarayıcınızda açılan sayfada sorularınızı yazın; uygulama bağlama göre cevap üretip gösterecektir.

7) Kod Hakkında Önemli Noktalar / Açıklamalar

load_embeddings() fonksiyonu: HuggingFaceEmbeddings modelini yüklüyor ve Streamlit cache ile tekrar yüklemeyi engelliyor.

build_vectorstore() fonksiyonu: metni parçalayarak FAISS üzerinde index oluşturuyor.

build_rag_chain() fonksiyonu:

Retriever oluşturur (k=4 ile en benzer 4 dökümanı alır).

HuggingFaceEndpoint ile generation modelini kullanır; huggingfacehub_api_token parametresi ile API token verilir.

Prompt: system + human mesajları ile chat-benzeri bir prompt şablonu oluşturulur.

app/main.py içinde with open("data/data.txt", "r", encoding="utf-8") şeklinde data yolunu güncellediğinizden emin olun.

8) Deploy / Yayınlama (Kısa Notlar)

Streamlit Community Cloud (eski adıyla streamlit.io deploy): Repo'yu bağlayıp streamlit run komutunu belirleyerek deploy edebilirsiniz. Secrets kısmına HUGGINGFACE_API_TOKEN eklemeyi unutmayın.

Hugging Face Spaces: Gradio veya Streamlit desteklenir; yine secret token eklenmelidir.

Diğer (Heroku, Docker vb.): Dockerfile oluşturup container içinde çalıştırabilirsiniz. (Heroku gibi servislerde secrets olarak token'ı tanımlayın.)

9) Değerlendirme & Sonuçlar

Beklenen çıktı: Kullanıcının sorusuna, vektör veritabanından getirilen ilgili bağlam parçalarını kullanarak LLM’in kısa ve doğru yanıtı.

Ölçümler:

Kalite: Doğruluk ve uydurma (fabrication) kontrolü — sistem prompt'unda "Bilmediğinde 'Elimdeki bağlamda bu bilgi yok' de" kuralı eklenmiştir.

Performans: FAISS ile retrieval hızlıdır; embedding oluşturma maliyeti model yüklemeye bağlıdır.

İyileştirme önerileri:

Daha büyük/kapsamlı veri seti,

Retrieval için semantic search parametrelerinin ayarlanması,

Daha sofistike prompt engineering,

Cevapların sayısal/istatistiksel olarak değerlendirilmesi (human eval).

10) Hata & Sorun Giderme

data.txt bulunamadı: data/data.txt yolunu kontrol edin.

Hugging Face token hatası: .streamlit/secrets.toml içine doğru token eklendiğinden emin olun.

ImportError/paket hatası: pip install -r requirements.txt ile tekrar kurun; Python sürümünüzü kontrol edin.

FAISS ile ilgili sistem uyumsuzluğu: işletim sistemi ve Python sürümü için faiss-cpu/faiss-gpu uyumunu kontrol edin.

11) Geliştirme Notları (İleri Adımlar)

Embeddingleri disk üzerinde saklayıp yeniden yükleme (index persist) ekleyin.

Daha gelişmiş RAG pipeline (ör. reranker, passage scoring) entegre edin.

Kullanıcı arayüzünü geliştirerek context highlight, kaynak gösterimi ekleyin.

12) Lisans & İletişim

Lisans: (tercihinize göre ekleyin, ör. MIT)

İletişim: Proje sahibi / geliştirici — e-posta veya GitHub profil linki (buraya ekleyin).
