'use client';

import { useState, useEffect } from 'react';
import { Check, Search, FileText, Sparkles, Target } from 'lucide-react';
import { ChronicCondition } from '@/types';
import { DataService } from '@/lib/dataService';

interface IcdCodeSelectionProps {
  condition: string;
  selectedIcdCode: string | null;
  onSelect: (icdCode: string, description: string) => void;
  suggestedIcdCode?: string;
  icdConfidence?: number;
  alternativeIcdCodes?: string[];
}

const IcdCodeSelection = ({ 
  condition, 
  selectedIcdCode, 
  onSelect, 
  suggestedIcdCode,
  icdConfidence,
  alternativeIcdCodes = []
}: IcdCodeSelectionProps) => {
  const [icdCodes, setIcdCodes] = useState<ChronicCondition[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    const codes = DataService.getIcdCodesForCondition(condition);
    setIcdCodes(codes);
    
    // Auto-select suggested ICD code if available and nothing is selected yet
    if (suggestedIcdCode && !selectedIcdCode && codes.length > 0) {
      const suggestedCode = codes.find(c => c.icdCode === suggestedIcdCode);
      if (suggestedCode) {
        onSelect(suggestedCode.icdCode, suggestedCode.icdDescription);
      }
    }
  }, [condition, suggestedIcdCode]);
  
  const filteredCodes = icdCodes.filter(code =>
    code.icdCode.toLowerCase().includes(searchTerm.toLowerCase()) ||
    code.icdDescription.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
          <FileText className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">ICD-10 Code Selection</h2>
          <p className="text-sm text-gray-500">Condition: {condition}</p>
        </div>
      </div>
      
      <div className="mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            className="input-field pl-10"
            placeholder="Search ICD codes or descriptions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>
      
      {suggestedIcdCode && icdConfidence && (
        <div className="mb-4 p-4 bg-gradient-to-r from-purple-50 to-blue-50 border-2 border-purple-200 rounded-lg">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 text-purple-600" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-purple-900 mb-1">AI Suggested ICD Code</h3>
              <p className="text-sm text-purple-700 mb-2">
                Based on your clinical note, we recommend: <span className="font-mono font-bold">{suggestedIcdCode}</span>
              </p>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-white rounded-full h-2 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                    style={{ width: `${Math.round(icdConfidence * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-purple-700">
                  {Math.round(icdConfidence * 100)}% confidence
                </span>
              </div>
              {alternativeIcdCodes.length > 0 && (
                <p className="text-xs text-purple-600 mt-2">
                  Alternative options: {alternativeIcdCodes.join(', ')}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
      
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {filteredCodes.map((code, index) => {
          const isSelected = selectedIcdCode === code.icdCode;
          const isSuggested = code.icdCode === suggestedIcdCode;
          const isAlternative = alternativeIcdCodes.includes(code.icdCode);
          
          return (
            <button
              key={`${code.icdCode}-${index}`}
              onClick={() => onSelect(code.icdCode, code.icdDescription)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? 'border-blue-500 bg-blue-50'
                  : isSuggested
                  ? 'border-purple-300 bg-purple-50 hover:border-purple-400'
                  : isAlternative
                  ? 'border-blue-200 bg-blue-50 hover:border-blue-300'
                  : 'border-gray-200 hover:border-blue-300 bg-white'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="font-mono font-semibold text-blue-600">
                      {code.icdCode}
                    </span>
                    {isSuggested && !isSelected && (
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        AI Suggested
                      </span>
                    )}
                    {isAlternative && !isSelected && !isSuggested && (
                      <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        Alternative
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-700">
                    {code.icdDescription}
                  </p>
                </div>
                
                {isSelected && (
                  <div className="ml-4 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
      
      {filteredCodes.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No ICD codes found matching your search.
        </div>
      )}
    </div>
  );
};

export default IcdCodeSelection;

