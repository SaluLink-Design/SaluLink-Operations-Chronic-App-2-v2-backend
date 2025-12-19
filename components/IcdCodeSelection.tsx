'use client';

import { useState, useEffect } from 'react';
import { Check, Search, FileText } from 'lucide-react';
import { ChronicCondition } from '@/types';
import { DataService } from '@/lib/dataService';

interface IcdCodeSelectionProps {
  condition: string;
  selectedIcdCode: string | null;
  onSelect: (icdCode: string, description: string) => void;
}

const IcdCodeSelection = ({ condition, selectedIcdCode, onSelect }: IcdCodeSelectionProps) => {
  const [icdCodes, setIcdCodes] = useState<ChronicCondition[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    const codes = DataService.getIcdCodesForCondition(condition);
    setIcdCodes(codes);
  }, [condition]);
  
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
      
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {filteredCodes.map((code, index) => {
          const isSelected = selectedIcdCode === code.icdCode;
          
          return (
            <button
              key={`${code.icdCode}-${index}`}
              onClick={() => onSelect(code.icdCode, code.icdDescription)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 bg-white'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono font-semibold text-blue-600">
                      {code.icdCode}
                    </span>
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

