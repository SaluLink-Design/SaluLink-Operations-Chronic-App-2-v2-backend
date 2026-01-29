'use client';

import { useState } from 'react';
import { Check, AlertCircle, ChevronDown, ChevronUp, Shield, Brain } from 'lucide-react';
import { MatchedCondition } from '@/types';

interface ConditionSelectionProps {
  matchedConditions: MatchedCondition[];
  onSelect: (condition: string, icdCode: string, description: string) => void;
  selectedCondition: string | null;
}

const ConditionSelection = ({ matchedConditions, onSelect, selectedCondition }: ConditionSelectionProps) => {
  const [showAll, setShowAll] = useState(false);
  const [expandedCondition, setExpandedCondition] = useState<string | null>(null);
  
  const displayedConditions = showAll ? matchedConditions : matchedConditions.slice(0, 5);
  
  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
          <AlertCircle className="w-6 h-6 text-primary-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Condition Identification</h2>
          <p className="text-sm text-gray-500">Select the most appropriate chronic condition</p>
        </div>
      </div>
      
      {matchedConditions.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No conditions matched. Try analyzing a different clinical note.</p>
        </div>
      ) : (
        <>
          <div className="space-y-3">
            {displayedConditions.map((condition, index) => {
              const isSelected = selectedCondition === condition.condition;
              const isExpanded = expandedCondition === `${condition.condition}-${index}`;
              const confidence = Math.round(condition.similarityScore * 100);
              const isConfirmed = condition.isConfirmed || false;
              const hasKeywords = condition.triggeringKeywords && condition.triggeringKeywords.length > 0;
              
              return (
                <div
                  key={`${condition.condition}-${condition.icdCode}-${index}`}
                  className={`rounded-lg border-2 transition-all ${
                    isSelected
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 bg-white'
                  }`}
                >
                  <button
                    onClick={() => onSelect(condition.condition, condition.icdCode, condition.icdDescription)}
                    className="w-full text-left p-4"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                          <h3 className="font-semibold text-lg text-gray-900">
                            {condition.condition}
                          </h3>
                          {isConfirmed && (
                            <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 flex items-center gap-1">
                              <Shield className="w-3 h-3" />
                              CONFIRMED
                            </span>
                          )}
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            confidence >= 80
                              ? 'bg-green-100 text-green-700'
                              : confidence >= 60
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-gray-100 text-gray-700'
                          }`}>
                            {confidence}% match
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-1">
                          ICD-10: <span className="font-medium">{condition.icdCode}</span>
                        </p>
                        <p className="text-sm text-gray-500 mb-2">
                          {condition.icdDescription}
                        </p>
                        {condition.matchExplanation && (
                          <p className="text-xs text-gray-500 italic flex items-center gap-1">
                            <Brain className="w-3 h-3" />
                            {condition.matchExplanation}
                          </p>
                        )}
                      </div>
                      
                      {isSelected && (
                        <div className="ml-4 w-6 h-6 bg-primary-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <Check className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </div>
                  </button>
                  
                  {hasKeywords && (
                    <div className="border-t border-gray-200 px-4 py-2">
                      <button
                        onClick={() => setExpandedCondition(isExpanded ? null : `${condition.condition}-${index}`)}
                        className="flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 w-full"
                      >
                        <span className="font-medium">
                          {isExpanded ? 'Hide' : 'Show'} Triggering Keywords
                        </span>
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                      
                      {isExpanded && (
                        <div className="mt-3 space-y-2">
                          <p className="text-xs text-gray-600 mb-2">
                            This condition was identified based on the following keywords from your clinical note:
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {condition.triggeringKeywords!.slice(0, 5).map((kw, kwIdx) => (
                              <div
                                key={kwIdx}
                                className="px-3 py-1 bg-blue-50 border border-blue-200 rounded-full"
                              >
                                <span className="text-sm font-medium text-blue-900">
                                  {kw.keyword}
                                </span>
                                <span className="text-xs text-blue-600 ml-1">
                                  ({Math.round(kw.similarityScore * 100)}%)
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          
          {matchedConditions.length > 5 && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="mt-4 w-full py-2 text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              {showAll ? 'Show Less' : `Show ${matchedConditions.length - 5} More`}
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default ConditionSelection;

