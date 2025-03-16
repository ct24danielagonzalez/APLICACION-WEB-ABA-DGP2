// Importar módulos de Firebase
import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js';
import { getAuth, createUserWithEmailAndPassword, sendEmailVerification } from 'https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js';

// Configuración de Firebase
const firebaseConfig = {
    apiKey: "AIzaSyDHHi3h2G5VY4bCg92QWZOjM6Bl6pco2ro",
    authDomain: "aba2025-8df97.firebaseapp.com",
    projectId: "aba2025-8df97",
    storageBucket: "aba2025-8df97.appspot.com",
    messagingSenderId: "819645801939",
    appId: "1:819645801939:web:2c4f13c8029003c434bb26"
};

// Inicializar Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Función para registrar usuario
function registerUser(event) {
    event.preventDefault();

    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const name = document.getElementById('register-name').value;

    if (!email || !password || !name) {
        alert("Por favor, completa todos los campos.");
        return;
    }

    createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            const user = userCredential.user;
            console.log('Usuario registrado:', user);

            // Enviar verificación de correo
            sendEmailVerification(user)
                .then(() => {
                    alert("Registro exitoso. Verifica tu correo electrónico antes de iniciar sesión.");
                })
                .catch((error) => {
                    console.error("Error al enviar verificación de correo:", error.message);
                });
        })
        .catch((error) => {
            console.error('Error al registrar usuario:', error.message);
            alert(`Error: ${error.message}`);
        });
}

// Asignar evento al botón de registro
document.getElementById('register-btn').addEventListener('click', registerUser);

document.getElementById('register-btn').addEventListener('click', function() {
    // Aquí puedes agregar validaciones o lógica adicional
    window.location.href = 'homepage.html'; // Redireccionar a la página homepage.html
});
