'use client';

import { useState } from 'react';
import { Loader2, Sparkles, CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { NoteQualityScore } from '@/types';

interface ClinicalNoteInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  noteQuality?: NoteQualityScore;
}

const ClinicalNoteInput = ({ value, onChange, onAnalyze, isAnalyzing, noteQuality }: ClinicalNoteInputProps) => {
  const [showQualityDetails, setShowQualityDetails] = useState(false);
  
  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-700 bg-green-50 border-green-200';
    if (score >= 50) return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    return 'text-red-700 bg-red-50 border-red-200';
  };
  
  const getQualityIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (score >= 50) return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
    return <XCircle className="w-5 h-5 text-red-600" />;
  };
  
  const getQualityLabel = (score: number) => {
    if (score >= 80) return 'High Quality';
    if (score >= 50) return 'Acceptable';
    return 'Needs Improvement';
  };
  
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-900">Clinical Note</h2>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Sparkles className="w-4 h-4" />
          <span>Powered by ClinicalBERT</span>
        </div>
      </div>
      
      <p className="text-gray-600 mb-4">
        Enter or paste the specialist's clinical notes below. The AI will analyze the text to identify potential chronic conditions.
      </p>
      
      <textarea
        className="textarea-field min-h-[300px]"
        placeholder="Enter clinical notes here...&#10;&#10;Example: Patient presents with persistent wheezing, shortness of breath, and chest tightness. History of allergic rhinitis. Symptoms worsen with exercise and cold air exposure..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={isAnalyzing}
      />
      
      {noteQuality && (
        <div className={`mt-4 border-2 rounded-lg p-4 ${getQualityColor(noteQuality.completenessScore)}`}>
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setShowQualityDetails(!showQualityDetails)}
          >
            <div className="flex items-center gap-3">
              {getQualityIcon(noteQuality.completenessScore)}
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-semibold">Note Quality: {getQualityLabel(noteQuality.completenessScore)}</span>
                  <span className="text-sm">({noteQuality.completenessScore}/100)</span>
                </div>
                <p className="text-sm mt-1">
                  {noteQuality.completenessScore >= 80 && "Comprehensive documentation with all key elements"}
                  {noteQuality.completenessScore >= 50 && noteQuality.completenessScore < 80 && "Good documentation, but some details could be added"}
                  {noteQuality.completenessScore < 50 && "Important clinical details may be missing"}
                </p>
              </div>
            </div>
            {showQualityDetails ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </div>
          
          {showQualityDetails && (noteQuality.warnings.length > 0 || noteQuality.missingElements.length > 0) && (
            <div className="mt-4 pt-4 border-t border-current border-opacity-20">
              {noteQuality.missingElements.length > 0 && (
                <div className="mb-3">
                  <h4 className="font-semibold text-sm mb-2">Missing Elements:</h4>
                  <ul className="text-sm space-y-1 ml-4">
                    {noteQuality.missingElements.map((element, idx) => (
                      <li key={idx} className="list-disc">{element}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {noteQuality.warnings.length > 0 && (
                <div>
                  <h4 className="font-semibold text-sm mb-2">Suggestions:</h4>
                  <ul className="text-sm space-y-1 ml-4">
                    {noteQuality.warnings.map((warning, idx) => (
                      <li key={idx} className="list-disc">{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      <div className="flex items-center justify-between mt-6">
        <div className="text-sm text-gray-500">
          {value.length} characters
        </div>
        
        <button
          className="btn-primary flex items-center gap-2"
          onClick={onAnalyze}
          disabled={!value.trim() || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Analyze Note</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ClinicalNoteInput;

