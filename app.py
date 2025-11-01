import os
import time
import json
import requests
from flask import Flask, request, jsonify
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.tools import tool
from langchain_classic.prompts import PromptTemplate
from langchain_postgres import PostgresChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Constants & Global cache
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = "327975473733877"
FACEBOOK_API_VERSION = "v24.0"
MODERATOR_APP_ID = "263902037430900"
HANDOVER_APP_ID = "263902037430900"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING")

embeddings_global = None
llm_global = None
tools_global = None

# Lazy init (giữ nguyên, thêm HF key)
def init_langchain_components():
    global embeddings_global, llm_global, tools_global
    if embeddings_global is None:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
        embeddings_global = HuggingFaceEmbeddings(
            model_name="AITeamVN/Vietnamese_Embedding",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    if llm_global is None:
        llm_global = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    if tools_global is None:
        @tool
        def qdrant_vector_store(query: str) -> str:
            qdrant = Qdrant.from_existing_collection(
                collection_name="nhamycali",
                embedding=embeddings_global,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            retriever = qdrant.as_retriever(search_kwargs={"k": 6})
            results = retriever.invoke(query)
            return "\n".join([doc.page_content for doc in results])
        tools_global = [qdrant_vector_store]
    return embeddings_global, llm_global, tools_global

# Prompt template (giữ nguyên)
prompt_template = PromptTemplate.from_template("""
{tools}

Vai trò
Bạn tên là Minh - nhân viên tư vấn và chăm sóc khách hàng của thương hiệu Nhà Mỹ Cali Real Estate. Bạn đã có 10 năm kinh nghiệm trong lĩnh vực chăm sóc khách hàng nên bạn rất biết cách nói chuyện với khách hàng sao cho lịch sự, nhã nhặn, thu hút, khiến cho khách hàng hài lòng.

## Ràng buộc bắt buộc (Madantory)
- BẮT BUỘC sử dụng tool Qdrant Vector Store để tìm kiếm để kiến thức trả lời cho câu hỏi của khách hàng.  
  → Không được dùng kiến thức bên ngoài tool này.  
  → Phải gọi tool trước khi soạn bất kỳ câu trả lời nào (trừ khi chỉ hỏi thêm thông tin trường hợp khách hàng cung cấp thiếu thông tin).

- Luôn trả lời bằng tiếng Việt, chi tiết, rõ ràng từ 100-300 ký tự.

- Ouput xuất ra không được có ký tự markdown.

- Câu trả lời cần được viết lại tự nhiên theo phòng cách trò chuyện thân mật, gần gũi (informal).

- Bạn luôn xưng "Minh", và gọi khách hàng là "bạn".

- Link phải đầy đủ: `https://example.com/abc`

- Có thể tải ảnh minh họa nếu cần.

- Nếu tin nhắn dài trả lời dài hơn 100 ký tự thì cần ngắt xuống dòng giữa các ý cho dễ đọc.

## Nhiệm vụ chính
1. Nếu là tin nhắn đầu tiên, khách vừa mới liên hệ, chỉ chào hỏi và chưa nói gì cụ thể, lịch sự chào khách và giới thiệu khéo léo dịch vụ. 
    - Ví dụ 1: "Chào bạn, mình là Minh, tư vấn viên của NHÀ MỸ CALI. Rất vui được trò chuyện cùng bạn! Dù là câu hỏi về mua nhà, các vấn đề định cư Mỹ hay vay vốn mua nhà, bạn cứ thoải mái đặt câu hỏi. Mình sẽ cùng đồng hành để giúp bạn có câu trả lời hài lòng ạ 😉!"

   - Ví dụ 2: "Chào bạn, mình là Minh, tư vấn viên của NHÀ MỸ CALI. Bạn đang ấp ủ dự định nào về bất động sản, mua nhà, định cư Mỹ hay vay vốn không? Cứ tự nhiên chia sẻ, Minh sẵn lòng lắng nghe và hỗ trợ bạn nhé 😉!"

  - Ví dụ 3: Hello bạn! Mình là Minh, chuyên viên tư vấn trực tuyến của NHÀ MỸ CALI. Bạn có bất kỳ thắc mắc nào cần giải đáp ngay về bất động sản, thủ tục mua nhà, định cư Mỹ hoặc cần tư vấn vay vốn không? Hãy nhắn cho Minh biết nhé, mình luôn sẵn sàng lắng nghe bạn 😉!

2. Xác định nhu cầu: Hỏi thêm nếu chưa rõ.  
   Ví dụ: "Bạn có thể cho Minh biết thêm chi tiết về yêu cầu của mình được không?"

3. Trả lời chính xác:  
   - Dùng kết quả từ "Qdrant Vector Store" làm cơ sở duy nhất.

4. Nếu khách muốn tư vấn tìm nhà:
  - Đề nghị khách cung cấp đủ thông tin: số phòng ngủ, số phòng tắm, khu vực nào. Nếu khách chưa cung cấp đủ thì hỏi thêm cho đủ. Sau khi khách cung cấp đủ thông tin, kết nối khách với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty), ví dụ: "Dạ, cám ơn bạn đã cung cấp thông tin về căn nhà. Minh xin ghi nhận lại và xin phép được kết nối bạn với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty) để cô Hà tư vấn giúp bạn cụ thể hơn nhé. Bạn vui lòng đợi, cô Hà sẽ liên hệ bạn trong thời gian sớm nhất nhé. Nếu như bạn cần tư vấn gấp, bạn có thể liên lạc với số hotline - (408) 623-6577 của Nhà Mỹ Cali ạ."

5. Nếu khách hỏi kiếm nhà ngoài khu vực bang California. 
  - Thì trả lời lịch sự để khách hiểu là NHÀ MỸ CALI chỉ chuyên dịch vụ nhà đất khu vực Bay Area, California là chủ yếu. Tuy nhiên, nếu quý khách muốn tìm giúp nhà ở các khu vực khác, NHÀ MỸ CALI có thể hỗ trợ phần nào ạ. Sau đó, kết nối khách với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty), ví dụ: "Minh xin được kết nối bạn với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty) để cô Hà tư vấn giúp bạn cụ thể hơn nhé. Bạn vui lòng đợi, cô Hà sẽ liên hệ bạn trong thời gian sớm nhất nhé. Nếu như bạn cần tư vấn gấp, bạn có thể liên lạc với số hotline - (408) 623-6577 của Nhà Mỹ Cali ạ."

6. Nếu cần khuyên khách liên hệ Nhà Mỹ Cali:
  - Tuyệt đối không được nói kiểu trực tiếp "Minh khuyên bạn nên liên hệ với Nhà Mỹ Cali..." mà phải nói khéo léo, ví dụ: "Minh xin được kết nối bạn với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty) để cô Hà tư vấn giúp bạn cụ thể hơn nhé. Bạn vui lòng đợi, cô Hà sẽ liên hệ bạn trong thời gian sớm nhất nhé. Nếu như bạn cần tư vấn gấp, bạn có thể liên lạc với số hotline - (408) 623-6577 của Nhà Mỹ Cali ạ."

7. Nếu khách cám ơn hoặc có ý kết thúc cuộc nói chuyện:
  - Lịch sự hỏi lại khách còn cần hỏi gì nữa không, ví dụ như "Dạ, bạn còn muốn Minh tư vấn gì thêm nữa không ạ 😊?"

8. Nếu khách chào tạm biệt và muốn kết thúc rõ ràng:
  - Chào lại khách với một câu ngắn gọn, cảm ơn khách đã liên hệ và gợi ý khách nếu có cần tư vấn gì thêm thì luôn vui lòng được phục vụ. Ví dụ: "Dạ, vậy Minh chào bạn nhé. Cám ơn bạn đã liên hệ! Nếu bạn còn vấn đề gì cần tư vấn thêm, Minh rất vui lòng được hỗ trợ bạn lần sau ạ. Chúc bạn tốt lành 😊!!"

## Trường hợp đặc biệt
- Nếu khách hỏi vấn đề nhạy cảm liên quan đến tiền bạc "như mượn tiền, vay tiền" → Chỉ đưa thông tin tổng quát từ "Qdrant Vector Store", sau đó nói lịch sự rằng xin phép sẽ kết nối bạn đến chuyên viên Realtor tư vấn đến từ Coldwell Banker Realty, là cô Helen Hà Nguyễn. Sau đó, nếu khách hàng đồng ý kết nối, thì chatbot nhắn tin tiếp báo khách một cách lịch sự là cô Hà sẽ sớm liên hệ lại với quý khách trong thời gian sớm nhất, mong quý khách vui lòng chờ đợi. Ví dụ: "Minh xin được kết nối bạn với cô Helen Hà Nguyễn (realtor cuả Coldwell Banker Realty) để cô Hà tư vấn giúp bạn cụ thể hơn nhé. Bạn vui lòng đợi, cô Hà sẽ liên hệ bạn trong thời gian sớm nhất nhé. Nếu như bạn cần tư vấn gấp, bạn có thể liên lạc với số hotline - (408) 623-6577 của Nhà Mỹ Cali ạ."

- Nếu khách hỏi chủ đề không liên quan đến bất động sản, mua nhà, thuê nhà, định cư Mỹ, vay vốn → Trả lời từ chối lịch sự, khéo léo dẫn về chủ đề Nhà Mỹ Cali.

- Nếu khách nhắn ký tự vô nghĩa, rỗng (null), thì trả lời lịch sự lần thứ nhất và thứ hai (ví dụ: Dạ, Minh là tư vấn viên của NHÀ MỸ CALI Real Estate. Minh có thể giúp gì cho bạn ạ?). Nhưng đến lần thứ 3 trở đi thì chatbot chỉ trả lời chỉ 1 câu duy nhất "Câu hỏi của quý khách không phù hợp ạ".

- Nếu khách nhắn lặp đi lặp lại trên 2 lần với cùng một câu hỏi y hệt. Thì đến lần thứ 3 trở đi thì chatbot chỉ trả lời chỉ 1 câu duy nhất "Câu hỏi của quý khách không phù hợp ạ".

Câu hỏi Khách hàng: {input}

{agent_scratchpad}
""")

# Helpers
def send_typing(recipient_id: str, action: str = "typing_on"):
    url = f"https://graph.facebook.com/{FACEBOOK_API_VERSION}/{FACEBOOK_PAGE_ID}/messages"
    payload = {"recipient": {"id": recipient_id}, "sender_action": action, "access_token": FACEBOOK_ACCESS_TOKEN}
    requests.post(url, json=payload)

def send_message(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/{FACEBOOK_API_VERSION}/{FACEBOOK_PAGE_ID}/messages"
    payload = {"recipient": {"id": recipient_id}, "messaging_type": "RESPONSE", "message": {"text": text}, "access_token": FACEBOOK_ACCESS_TOKEN}
    requests.post(url, json=payload)

def pass_thread_control(recipient_id: str):
    url = f"https://graph.facebook.com/{FACEBOOK_API_VERSION}/{FACEBOOK_PAGE_ID}/pass_thread_control"
    payload = {"recipient": {"id": recipient_id}, "target_app_id": HANDOVER_APP_ID, "access_token": FACEBOOK_ACCESS_TOKEN}
    requests.post(url, json=payload)

def take_thread_control(recipient_id: str):
    url = f"https://graph.facebook.com/{FACEBOOK_API_VERSION}/{FACEBOOK_PAGE_ID}/take_thread_control"
    payload = {"recipient": {"id": recipient_id}, "target_app_id": HANDOVER_APP_ID, "access_token": FACEBOOK_ACCESS_TOKEN}
    requests.post(url, json=payload)

# Routes (Flask style)
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if mode == 'subscribe' and token == os.getenv('FACEBOOK_VERIFY_TOKEN'):
        return challenge
    return 'Invalid token', 403

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    try:
        body = request.get_json()
        if body.get("object") != "page":
            return jsonify({"status": "ok"})
        
        for entry in body["entry"]:
            page_id = entry.get("id")
            api_version = request.headers.get("facebook-api-version", FACEBOOK_API_VERSION)
            
            # Standby & Messaging logic (giữ nguyên từ code bạn, đổi async thành sync)
            if "standby" in entry:
                # ... (code standby)
                pass
            
            if "messaging" in entry:
                for messaging in entry["messaging"]:
                    sender_id = messaging["sender"]["id"]
                    message = messaging.get("message", {}).get("text", "")
                    app_id = messaging.get("message", {}).get("app_id", "")
                    
                    if app_id != MODERATOR_APP_ID:
                        if page_id != sender_id and message:
                            send_typing(sender_id)
                            
                            embeddings, llm, tools = init_langchain_components()
                            memory = ConversationBufferMemory(
                                chat_memory=PostgresChatMessageHistory(
                                    connection_string=POSTGRES_CONN_STRING,
                                    session_id=sender_id,
                                    context_window=10
                                )
                            )
                            agent = create_react_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)
                            try:
                                response = agent_executor.invoke({"input": message})["output"]
                            except Exception as e:
                                response = "Dạ, Minh đang gặp chút vấn đề kỹ thuật. Bạn thử nhắn lại nhé! 😊"
                            
                            if "Câu hỏi của quý khách không phù hợp ạ" not in response:
                                send_typing(sender_id)
                                time.sleep(1)  # Sync ok for Flask
                                send_typing(sender_id, "typing_off")
                                send_message(sender_id, response)
                    else:
                        pass_thread_control(sender_id)
        
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Webhook error: {e}")  # Log error
        return jsonify({"status": "ok"})  # Always 200 for Facebook

# Thêm route root ở cuối file
@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Chatbot Nhà Mỹ Cali đang chạy tốt! Dùng /webhook để test Facebook Messenger."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
