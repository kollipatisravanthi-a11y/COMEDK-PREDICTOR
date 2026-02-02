function toggleChat() {
    const chatWidget = document.getElementById('chat-widget');
    const toggleIcon = document.getElementById('chat-toggle-icon');
    
    chatWidget.classList.toggle('expanded');
    
    if (chatWidget.classList.contains('expanded')) {
        toggleIcon.classList.remove('fa-chevron-up');
        toggleIcon.classList.add('fa-chevron-down');
        document.getElementById('chat-input').focus();
    } else {
        toggleIcon.classList.remove('fa-chevron-down');
        toggleIcon.classList.add('fa-chevron-up');
    }
}

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const message = inputField.value.trim();
    
    if (message === '') return;
    
    // Add user message
    addMessage(message, 'user-message');
    inputField.value = '';
    
    // Show typing indicator (optional, but nice)
    // ...
    
    // Send to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response, 'bot-message');
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage('Sorry, something went wrong. Please try again.', 'bot-message');
    });
}

function addMessage(text, className) {
    const chatBody = document.getElementById('chat-body');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', className);
    messageDiv.textContent = text;
    
    chatBody.appendChild(messageDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function clearChat() {
    const chatBody = document.getElementById('chat-body');
    chatBody.innerHTML = '';
    
    // Restore initial greeting
    const initialMessage = document.createElement('div');
    initialMessage.classList.add('chat-message', 'bot-message');
    initialMessage.textContent = "Hello! I'm here to help you with COMEDK related queries. Ask me anything!";
    chatBody.appendChild(initialMessage);
}
