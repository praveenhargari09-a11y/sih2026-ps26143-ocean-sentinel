import { createContext, useContext } from 'react'
import type { LayerVisibility } from '../types'

interface LayerCtx {
  layers: LayerVisibility
  toggle: (k: keyof LayerVisibility) => void
}

export const LayerContext = createContext<LayerCtx>({
  layers: {
    spillPolygon: true,
    hindcast: true,
    forecast: true,
    vesselTracks: true,
    uncertaintyCircle: true,
  },
  toggle: () => {},
})

export const useLayerContext = () => useContext(LayerContext)
