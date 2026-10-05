import React from 'react';

export interface DataSourceBadgeProps {
  aisSource?: string;
  sarSource?: string;
}

function getSourceDisplay(source?: string) {
  if (!source) return { label: 'Unknown', colorClass: 'bg-ops-muted text-ops-panel border-ops-muted' };
  
  const lower = source.toLowerCase();
  if (lower.includes('real_with_placeholders') || lower === 'real') {
    return { label: 'Real', colorClass: 'bg-success/20 text-success border-success/40' };
  }
  if (lower.includes('reconstructed')) {
    return { label: 'Reconstructed from public records', colorClass: 'bg-amber-500/20 text-amber-400 border-amber-400/40' };
  }
  if (lower.includes('synthetic')) {
    return { label: 'Synthetic placeholder', colorClass: 'bg-ops-subtle text-ops-muted border-ops-border' };
  }
  return { label: source, colorClass: 'bg-ops-subtle text-ops-muted border-ops-border' };
}

export default function DataSourceBadge({ aisSource, sarSource }: DataSourceBadgeProps) {
  if (!aisSource && !sarSource) return null;

  const aisDisplay = getSourceDisplay(aisSource);
  const sarDisplay = getSourceDisplay(sarSource);

  return (
    <div className="flex items-center gap-2">
      <div 
        className={`px-2 py-0.5 rounded text-[10px] font-medium border flex items-center gap-1.5 ${sarDisplay.colorClass}`}
        title={`SAR Source: ${sarSource || 'Unknown'}`}
      >
        <span className="opacity-75">SAR:</span> {sarDisplay.label}
      </div>
      <div 
        className={`px-2 py-0.5 rounded text-[10px] font-medium border flex items-center gap-1.5 ${aisDisplay.colorClass}`}
        title={`AIS Source: ${aisSource || 'Unknown'}`}
      >
        <span className="opacity-75">AIS:</span> {aisDisplay.label}
      </div>
    </div>
  );
}
