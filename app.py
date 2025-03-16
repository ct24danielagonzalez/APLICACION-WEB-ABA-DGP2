from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# Cargar las variables desde el archivo api.env
load_dotenv('api.env')
groq_api_key = os.getenv("GROQ_API_KEY")

class GroqChatClient:
    def __init__(self, model_id='llama-3.1-8b-instant', system_message=None, api_key=None):
        # Usar la clave API cargada desde las variables de entorno
        if api_key is None:
            api_key = groq_api_key  # Asignar la clave API desde las variables de entorno si no se proporciona

        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError("No se proporcionó una clave API válida")
        
        self.model_id = model_id
        self.messages = []

        if system_message:
            self.messages.append({'role': 'system', 'content': system_message})

    def draft_message(self, prompt, role='user'):
        return {'role': role, 'content': prompt}

    def send_request(self, message, temperature=0.5, max_tokens=1024, stream=False, stop=None):
        self.messages.append(message)

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop
        )

        if not stream:
            response = {
                'content': chat_completion.choices[0].message.content,
                'finish_reason': chat_completion.choices[0].finish_reason,
                'role': chat_completion.choices[0].message.role,
                'prompt_tokens': chat_completion.usage.prompt_tokens,
                'prompt_time': chat_completion.usage.prompt_time,
                'completion_tokens': chat_completion.usage.completion_tokens,
                'completion_time': chat_completion.usage.completion_time,
                'total_tokens': chat_completion.usage.total_tokens,
                'total_time': chat_completion.usage.total_time
            }
            self.messages.append(self.draft_message(response['content'], response['role']))
            return response
        return chat_completion

    @property
    def last_message(self):
        return self.messages[-1]

# Inicializar el cliente de chat
system_message = """Tu eres un terapeuta, psicologo, psiquiatra y un apoyo emocional, especializado en el tratamiento y manejo de trastornos como la depresión, ansiedad, problemas de autoestima, trastornos alimenticios y emociones difíciles de gestionar. 
Con la experiencia y empatía de un terapeuta, psicólogo y psiquiatra con 10 años de formación en salud mental, ABA busca ofrecer un espacio seguro y confidencial para las personas que buscan ayuda en momentos de vulnerabilidad emocional. 
Su objetivo es proporcionar respuestas adecuadas para cada situación, ayudando a las personas a reconocer sus emociones, promover el bienestar mental y fomentar un ambiente de apoyo en la gestión de trastornos emocionales.""".strip().replace("\n", "")

client = GroqChatClient(system_message=system_message)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    
    response = client.send_request(client.draft_message(user_message), stream=False)
    
    return jsonify({
        'content': response['content'],
        'role': response['role']
    })

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)