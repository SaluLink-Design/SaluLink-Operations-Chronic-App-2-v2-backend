'use client';

import { useState, useEffect } from 'react';
import { Plus, Trash2, Upload, FileText } from 'lucide-react';
import { TreatmentBasketItem, TreatmentItem } from '@/types';
import { DataService } from '@/lib/dataService';

interface DiagnosticBasketProps {
  condition: string;
  treatments: TreatmentItem[];
  onAddTreatment: (treatment: TreatmentItem) => void;
  onUpdateTreatment: (index: number, treatment: Partial<TreatmentItem>) => void;
  onRemoveTreatment: (index: number) => void;
}

const DiagnosticBasket = ({
  condition,
  treatments,
  onAddTreatment,
  onUpdateTreatment,
  onRemoveTreatment
}: DiagnosticBasketProps) => {
  const [basketItems, setBasketItems] = useState<TreatmentBasketItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  
  useEffect(() => {
    const items = DataService.getTreatmentBasketForCondition(condition);
    setBasketItems(items);
  }, [condition]);
  
  const handleAddTreatment = () => {
    if (!selectedItem) return;
    
    const item = basketItems.find(b => b.diagnosticBasket.description === selectedItem);
    if (!item) return;
    
    const newTreatment: TreatmentItem = {
      description: item.diagnosticBasket.description,
      code: item.diagnosticBasket.code,
      maxCovered: parseInt(item.diagnosticBasket.covered) || 1,
      timesCompleted: 1,
      documentation: {
        notes: '',
        images: []
      }
    };
    
    onAddTreatment(newTreatment);
    setSelectedItem(null);
  };
  
  const availableItems = basketItems.filter(item =>
    !treatments.some(t => t.description === item.diagnosticBasket.description) &&
    item.diagnosticBasket.description
  );
  
  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
          <FileText className="w-6 h-6 text-green-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Diagnostic Basket</h2>
          <p className="text-sm text-gray-500">Select and document required diagnostic tests</p>
        </div>
      </div>
      
      {/* Add Treatment */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <label className="label">Add Diagnostic Test</label>
        <div className="flex gap-2">
          <select
            className="input-field flex-1"
            value={selectedItem || ''}
            onChange={(e) => setSelectedItem(e.target.value)}
          >
            <option value="">Select a diagnostic test...</option>
            {availableItems.map((item, index) => (
              <option key={index} value={item.diagnosticBasket.description}>
                {item.diagnosticBasket.description} ({item.diagnosticBasket.code}) - Max: {item.diagnosticBasket.covered}
              </option>
            ))}
          </select>
          <button
            className="btn-primary"
            onClick={handleAddTreatment}
            disabled={!selectedItem}
          >
            <Plus className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Treatment List */}
      <div className="space-y-4">
        {treatments.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No diagnostic tests added yet</p>
          </div>
        ) : (
          treatments.map((treatment, index) => (
            <div key={index} className="border-2 border-gray-200 rounded-lg p-4">
              {/* Treatment Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-gray-900 mb-1">
                    {treatment.description}
                  </h3>
                  <p className="text-sm text-gray-600">
                    Code: <span className="font-mono">{treatment.code}</span>
                  </p>
                </div>
                <button
                  onClick={() => onRemoveTreatment(index)}
                  className="text-red-600 hover:text-red-700 p-2"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
              
              {/* Times Completed */}
              <div className="mb-4">
                <label className="label">Times Completed</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="1"
                    max={treatment.maxCovered}
                    value={treatment.timesCompleted}
                    onChange={(e) =>
                      onUpdateTreatment(index, {
                        timesCompleted: Math.min(
                          parseInt(e.target.value) || 1,
                          treatment.maxCovered
                        )
                      })
                    }
                    className="input-field w-24"
                  />
                  <span className="text-sm text-gray-600">
                    of {treatment.maxCovered} covered
                  </span>
                </div>
              </div>
              
              {/* Documentation */}
              <div className="space-y-3">
                <label className="label">Documentation</label>
                
                <textarea
                  className="textarea-field"
                  rows={3}
                  placeholder="Enter findings and results..."
                  value={treatment.documentation.notes}
                  onChange={(e) =>
                    onUpdateTreatment(index, {
                      documentation: {
                        ...treatment.documentation,
                        notes: e.target.value
                      }
                    })
                  }
                />
                
                <div className="flex items-center gap-2">
                  <button className="btn-secondary text-sm flex items-center gap-2">
                    <Upload className="w-4 h-4" />
                    Upload Images
                  </button>
                  {treatment.documentation.images.length > 0 && (
                    <span className="text-sm text-gray-600">
                      {treatment.documentation.images.length} file(s) uploaded
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default DiagnosticBasket;

