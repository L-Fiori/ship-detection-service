import React, { useEffect, useState} from 'react';
import './ProcessingPage.css';

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const ProcessingPage = () => {
  useEffect(() => {
    fetch('http://localhost:3000/processing')
    .then((data) => {
      console.log(data.image)
    })
  })

  return (
    <div className="processing-container">
      <h2 className="processing-message">Processing Your Data...</h2>
      <div className="spinner"></div>
    </div>
  );
};

export default ProcessingPage;

