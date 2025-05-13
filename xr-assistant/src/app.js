import React, { useState } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    setResponse('');

    try {
      const res = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse('Error: Could not fetch response.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>XR Onboarding Assistant</h1>
      <textarea
        rows="4"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask your question here..."
      />
      <button onClick={sendMessage} disabled={loading}>
        {loading ? 'Thinking...' : 'Ask Assistant'}
      </button>
      <div className="response-box">
        {loading ? <div className="loader" /> : <p>{response}</p>}
      </div>
    </div>
  );
}

export default App;
