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

function formatMessage(text) {
    // 1. Escape HTML
    let formatted = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // 2. Bold (**text**)
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 3. Markdown Links [text](url)
    formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // 4. Raw URLs (http/s) that weren't captured by markdown
    // Avoid double-linking by checking if it's already inside an anchor tag
    // Simple lookbehind alternative or just skipping raw url if we assume backend is good.
    // Let's add simple raw URL support for fallback:
    formatted = formatted.replace(/(?<!href=")(https?:\/\/[^\s]+)/g, (match) => {
        // If it's already part of the markdown replacement (which we did first), this regex might overlap? 
        // Actually, step 3 handled [text](url). If we have https://google.com it remains.
        return `<a href="${match}" target="_blank">${match}</a>`;
    }); 
    
    // 5. Newlines
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function addMessage(text, className) {
    const chatBody = document.getElementById('chat-body');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', className);
    
    // Use innerHTML to render formatting
    messageDiv.innerHTML = formatMessage(text);
    
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
