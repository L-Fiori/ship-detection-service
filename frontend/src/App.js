import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import InputPage from './InputPage';
import ProcessingPage from './ProcessingPage';
import ResultPage from './ResultPage';

function App() {
  const handleInputSubmit = (formData) => {
    // Logic to handle the form data
    console.log('Form Data:', formData);
    // You might want to navigate to another route after form submission
    // For example: history.push('/processing');
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route exact path="/" element={<InputPage onSubmit={handleInputSubmit} />} />
        <Route path="/processing" element={<ProcessingPage />} />
        {<Route path="/result_page" element={<ResultPage />} />}
        {/* Add other routes as needed */}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
