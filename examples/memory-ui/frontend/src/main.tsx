import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/global.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  // StrictMode double-invokes effects in dev. That is deliberate here: it
  // surfaces a stream or browse that isn't properly cancelled on cleanup, which
  // is exactly the class of bug that would leak a previous user's memories into
  // the panel mid-demo.
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
