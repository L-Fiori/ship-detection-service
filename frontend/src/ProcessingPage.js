import React from 'react';
import './ProcessingPage.css';

const ProcessingPage = () => {
  return (
    <div className="processing-container">
      <h2 className="processing-message">Processing Your Data...</h2>
      <div className="spinner"></div>
    </div>
  );
};

export default ProcessingPage;

