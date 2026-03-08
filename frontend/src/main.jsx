import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

// Apply saved theme immediately to avoid flash
const saved = localStorage.getItem('snapclass-theme') || 'dark';
document.documentElement.dataset.theme = saved;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
