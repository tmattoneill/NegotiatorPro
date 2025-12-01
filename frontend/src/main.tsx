import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import LandingPage from './components/LandingPage.tsx'
import Login from './components/Login.tsx'
import AdminLogin from './components/AdminLogin.tsx'
import CreateProfile from './components/CreateProfile.tsx'
import UserProfile from './components/UserProfile.tsx'
import PersonaManager from './components/PersonaManager.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<App />} />
        <Route path="/login" element={<Login />} />
        <Route path="/admin-login" element={<AdminLogin />} />
        <Route path="/create-profile" element={<CreateProfile />} />
        <Route path="/profile" element={<UserProfile />} />
        <Route path="/personas" element={<PersonaManager />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
