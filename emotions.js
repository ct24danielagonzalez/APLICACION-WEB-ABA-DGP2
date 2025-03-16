function selectEmotion(emotion) {
    const messageElement = document.getElementById('recommendation-message');
    let activityName = '';
    
    switch(emotion) {
        case 'feliz':
            activityName = 'VÍNCULOS VALIOSOS';
            break;
        case 'triste':
            activityName = 'CALMA INTERIOR';
            break;
        case 'enojado':
            activityName = 'EXPRESIÓN CREATIVA';
            break;
        case 'emocionado':
            activityName = 'MOVIMIENTO VITAL';
            break;
    }
    
    messageElement.innerHTML = `
        Te recomendamos intentar <strong>${activityName}</strong>.<br>
        <a href="activities.html" class="explore-link">Explora esta y más actividades en nuestra sección de recomendaciones →</a>
    `;
    
    messageElement.classList.remove('visible');
    void messageElement.offsetWidth; 
    messageElement.classList.add('visible');
}