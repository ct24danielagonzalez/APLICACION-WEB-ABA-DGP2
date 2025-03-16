import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.2.0/firebase-app.js';
import { getAuth, signInWithEmailAndPassword, onAuthStateChanged } from 'https://www.gstatic.com/firebasejs/11.2.0/firebase-auth.js';

// Configuración de Firebase
const firebaseConfig = {
    apiKey: "AIzaSyDHHi3h2G5VY4bCg92QWZOjM6Bl6pco2ro",
    authDomain: "aba2025-8df97.firebaseapp.com",
    projectId: "aba2025-8df97",
    storageBucket: "aba2025-8df97.firebasestorage.app",
    messagingSenderId: "819645801939",
    appId: "1:819645801939:web:2c4f13c8029003c434bb26"
};

// Inicializar Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Función para iniciar sesión
function loginUser(event) {
    event.preventDefault(); // Evitar que el formulario se envíe automáticamente

    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    signInWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Inicio de sesión exitoso
            const user = userCredential.user;
            console.log('Usuario autenticado:', user);
            // Redirigir al usuario a la página principal después de iniciar sesión
            window.location.href = 'homepage.html';
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error('Error al iniciar sesión:', errorCode, errorMessage);
            alert('Error al iniciar sesión: ' + errorMessage);
        });
}

// Escuchar cambios en el estado de autenticación
onAuthStateChanged(auth, (user) => {
    if (user) {
        // Usuario está autenticado
        console.log('Usuario autenticado:', user);
    } else {
        // Usuario no está autenticado
        console.log('No hay usuario autenticado');
    }
});

// Asignar evento al formulario de inicio de sesión
document.getElementById('formulario-sesion').addEventListener('submit', loginUser);
