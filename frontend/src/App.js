// src/App.js
import React from 'react';
import InputPage from './InputPage';

function App() {
  const handleInputSubmit = (formData) => {
    // You can add logic to handle the form data (e.g., send it to the server)
    console.log('Form Data:', formData);
  };

  return (
    <div className="App">
      <InputPage onSubmit={handleInputSubmit} />
    </div>
  );
}

export default App;

