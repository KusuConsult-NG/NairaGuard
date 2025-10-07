import React, { createContext, useContext, useReducer, ReactNode } from 'react';

// Types
export interface DetectionResult {
  predicted_class: 'genuine' | 'fake';
  confidence: number;
  probabilities: number[];
  timestamp: string;
  model_status: string;
  filename?: string;
}

export interface AppState {
  detectionResult: DetectionResult | null;
  uploadedImage: File | null;
  imagePreview: string | null;
  isLoading: boolean;
  error: string | null;
  detectionHistory: DetectionResult[];
  modelStatus: {
    model_loaded: boolean;
    preprocessor_loaded: boolean;
    timestamp: string;
  } | null;
}

export interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

// Action types
export type AppAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_DETECTION_RESULT'; payload: DetectionResult }
  | { type: 'SET_UPLOADED_IMAGE'; payload: { file: File; preview: string } }
  | { type: 'CLEAR_DETECTION' }
  | { type: 'ADD_TO_HISTORY'; payload: DetectionResult }
  | { type: 'SET_MODEL_STATUS'; payload: AppState['modelStatus'] }
  | { type: 'RESET_STATE' };

// Initial state
const initialState: AppState = {
  detectionResult: null,
  uploadedImage: null,
  imagePreview: null,
  isLoading: false,
  error: null,
  detectionHistory: [],
  modelStatus: null,
};

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };

    case 'SET_DETECTION_RESULT':
      return {
        ...state,
        detectionResult: action.payload,
        isLoading: false,
        error: null,
      };

    case 'SET_UPLOADED_IMAGE':
      return {
        ...state,
        uploadedImage: action.payload.file,
        imagePreview: action.payload.preview,
        error: null,
      };

    case 'CLEAR_DETECTION':
      return {
        ...state,
        detectionResult: null,
        uploadedImage: null,
        imagePreview: null,
        error: null,
      };

    case 'ADD_TO_HISTORY':
      return {
        ...state,
        detectionHistory: [
          action.payload,
          ...state.detectionHistory.slice(0, 49),
        ], // Keep last 50
      };

    case 'SET_MODEL_STATUS':
      return { ...state, modelStatus: action.payload };

    case 'RESET_STATE':
      return initialState;

    default:
      return state;
  }
}

// Context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

// Action creators
export const appActions = {
  setLoading: (loading: boolean): AppAction => ({
    type: 'SET_LOADING',
    payload: loading,
  }),

  setError: (error: string | null): AppAction => ({
    type: 'SET_ERROR',
    payload: error,
  }),

  setDetectionResult: (result: DetectionResult): AppAction => ({
    type: 'SET_DETECTION_RESULT',
    payload: result,
  }),

  setUploadedImage: (file: File, preview: string): AppAction => ({
    type: 'SET_UPLOADED_IMAGE',
    payload: { file, preview },
  }),

  clearDetection: (): AppAction => ({
    type: 'CLEAR_DETECTION',
  }),

  addToHistory: (result: DetectionResult): AppAction => ({
    type: 'ADD_TO_HISTORY',
    payload: result,
  }),

  setModelStatus: (status: AppState['modelStatus']): AppAction => ({
    type: 'SET_MODEL_STATUS',
    payload: status,
  }),

  resetState: (): AppAction => ({
    type: 'RESET_STATE',
  }),
};
