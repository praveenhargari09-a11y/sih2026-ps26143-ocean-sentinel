/**
 * UploadPanel — drag-and-drop SAR file upload modal.
 * Clean, minimal — no glassmorphism, no decorative elements.
 */
import React, { useCallback, useState, useRef } from 'react'
import { Upload, X, FileImage, AlertTriangle, CheckCircle } from 'lucide-react'

interface Props {
  isOpen: boolean
  onClose: () => void
  onUpload: (formData: FormData) => Promise<void>
}

type DropState = 'idle' | 'dragging' | 'uploading' | 'success' | 'error'

export default function UploadPanel({ isOpen, onClose, onUpload }: Props) {
  const [dropState, setDropState] = useState<DropState>('idle')
  const [file, setFile]           = useState<File | null>(null)
  const [errorMsg, setErrorMsg]   = useState<string>('')
  const inputRef                  = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((f: File | null) => {
    if (!f) return
    if (!f.name.match(/\.(tif|tiff|nc|zip)$/i)) {
      setErrorMsg('Expected GeoTIFF (.tif/.tiff), NetCDF (.nc), or ZIP')
      setDropState('error')
      return
    }
    setFile(f)
    setDropState('idle')
    setErrorMsg('')
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDropState('idle')
    handleFile(e.dataTransfer.files[0] ?? null)
  }, [handleFile])

  const handleSubmit = useCallback(async () => {
    if (!file) return
    setDropState('uploading')
    const fd = new FormData()
    fd.append('file', file)
    try {
      await onUpload(fd)
      setDropState('success')
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Upload failed')
      setDropState('error')
    }
  }, [file, onUpload])

  if (!isOpen) return null

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(2px)' }}
      onClick={onClose}
    >
      {/* Modal */}
      <div
        className="relative w-[480px] rounded-lg shadow-2xl"
        style={{ background: '#161b22', border: '1px solid #30363d' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-ops-border">
          <div>
            <div className="text-[13px] font-semibold text-[#e6edf3]">Upload SAR Scene</div>
            <div className="text-[11px] text-ops-muted mt-0.5">GeoTIFF, NetCDF, or ZIP archive</div>
          </div>
          <button
            onClick={onClose}
            className="text-ops-muted hover:text-[#e6edf3] rounded p-1 transition-colors"
          ><X size={16} /></button>
        </div>

        {/* Drop zone */}
        <div className="px-5 py-5">
          <div
            onDragOver={e => { e.preventDefault(); setDropState('dragging') }}
            onDragLeave={() => setDropState('idle')}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`flex flex-col items-center justify-center gap-3 h-44 rounded cursor-pointer
                         border-2 border-dashed transition-colors
                         ${dropState === 'dragging' ? 'border-cyan-400 bg-cyan-400/5'
                           : dropState === 'error'   ? 'border-danger/50 bg-danger/5'
                           : dropState === 'success' ? 'border-success/50 bg-success/5'
                           : 'border-ops-border hover:border-ops-muted'}`}
          >
            {dropState === 'uploading' ? (
              <>
                <div className="w-8 h-8 border-2 border-ops-border border-t-amber-400 rounded-full animate-spin" />
                <div className="text-[12px] text-ops-muted">Processing SAR scene…</div>
              </>
            ) : dropState === 'success' ? (
              <>
                <CheckCircle size={28} className="text-success" />
                <div className="text-[13px] text-[#e6edf3]">Detection complete</div>
              </>
            ) : dropState === 'error' ? (
              <>
                <AlertTriangle size={28} className="text-danger" />
                <div className="text-[13px] text-danger">{errorMsg}</div>
                <div className="text-[11px] text-ops-muted">Click to try again</div>
              </>
            ) : file ? (
              <>
                <FileImage size={28} className="text-cyan-400" />
                <div className="text-[13px] text-[#e6edf3] font-medium">{file.name}</div>
                <div className="text-[11px] text-ops-muted">{(file.size / 1024 / 1024).toFixed(1)} MB · Ready to process</div>
              </>
            ) : (
              <>
                <Upload size={28} className="text-ops-muted" />
                <div className="text-[13px] text-[#e6edf3]">Drop SAR file here</div>
                <div className="text-[11px] text-ops-muted">.tif · .tiff · .nc · .zip</div>
              </>
            )}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept=".tif,.tiff,.nc,.zip"
            className="hidden"
            onChange={e => handleFile(e.target.files?.[0] ?? null)}
          />
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 pb-4 gap-3">
          <div className="text-[11px] text-ops-muted">
            SAR scene will be processed through the detection pipeline
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-[12px] rounded border border-ops-border
                         text-ops-muted hover:text-[#e6edf3] hover:border-ops-muted transition-colors"
            >Cancel</button>
            <button
              onClick={handleSubmit}
              disabled={!file || dropState === 'uploading'}
              className="px-4 py-1.5 text-[12px] font-medium rounded transition-colors
                         bg-amber-500/10 border border-amber-400/40 text-amber-400
                         hover:bg-amber-500/20 hover:border-amber-400/60
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {dropState === 'uploading' ? 'Processing…' : 'Run Detection'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
