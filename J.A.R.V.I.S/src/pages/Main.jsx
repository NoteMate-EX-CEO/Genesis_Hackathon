import React, { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

export default function Main(){
  const iframeRef = useRef(null)
  const [token, setToken] = useState('')
  const navigate = useNavigate()

  useEffect(()=>{
    const t = localStorage.getItem('token')||''
    if(!t){
      navigate('/login')
      return
    }
    setToken(t)
  },[])

  const demoUrl = token ? `http://localhost:8000/demo?token=${encodeURIComponent(token)}` : 'about:blank'

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <header className="sticky top-0 z-10 border-b border-gray-800 bg-black/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 border-2 border-[#7A0000] rounded-lg flex items-center justify-center font-bold">JR</div>
            <div className="tracking-widest font-extrabold">J.A.R.V.I.S</div>
            <span className="text-gray-500 text-sm ml-3">Enterprise RAG</span>
          </div>
          <nav className="flex items-center gap-3">
            <Link to="/" className="text-gray-300 hover:text-white">Home</Link>
            <Link to="/interviewer" className="text-gray-300 hover:text-white">AI Interviewer</Link>
            <Link to="/login" className="text-gray-300 hover:text-white">Switch User</Link>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="mb-3 text-sm text-gray-400">Logged in. Your chat runs below with black/red theme.</div>
          <div className="rounded-xl overflow-hidden border border-gray-800 shadow-2xl" style={{height:'78vh'}}>
            <iframe ref={iframeRef} title="RAG Demo" src={demoUrl} className="w-full h-full bg-black" />
          </div>
        </div>
      </main>
    </div>
  )
}
