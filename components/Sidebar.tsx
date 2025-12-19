'use client';

import { X, FolderOpen, Clock, CheckCircle } from 'lucide-react';
import { PatientCase } from '@/types';
import { format } from 'date-fns';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  cases: PatientCase[];
  onLoadCase: (caseId: string) => void;
  onDeleteCase: (caseId: string) => void;
}

const Sidebar = ({ isOpen, onClose, cases, onLoadCase, onDeleteCase }: SidebarProps) => {
  const getStatusColor = (status: PatientCase['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-700';
      case 'ongoing':
        return 'bg-blue-100 text-blue-700';
      case 'diagnostic':
        return 'bg-yellow-100 text-yellow-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };
  
  const getStatusIcon = (status: PatientCase['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4" />;
      case 'ongoing':
        return <Clock className="w-4 h-4" />;
      default:
        return <FolderOpen className="w-4 h-4" />;
    }
  };
  
  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full w-96 bg-white shadow-2xl z-50 transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div>
              <h2 className="text-xl font-bold text-gray-900">Patient Cases</h2>
              <p className="text-sm text-gray-500">{cases.length} case(s)</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
          
          {/* Cases List */}
          <div className="flex-1 overflow-y-auto p-4">
            {cases.length === 0 ? (
              <div className="text-center py-12">
                <FolderOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No saved cases yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {cases.map(patientCase => (
                  <div
                    key={patientCase.id}
                    className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors cursor-pointer"
                    onClick={() => {
                      onLoadCase(patientCase.id);
                      onClose();
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900">
                          {patientCase.patientName}
                        </h3>
                        <p className="text-sm text-gray-600">
                          ID: {patientCase.patientId}
                        </p>
                      </div>
                      <div className={`px-2 py-1 rounded flex items-center gap-1 text-xs font-medium ${getStatusColor(patientCase.status)}`}>
                        {getStatusIcon(patientCase.status)}
                        <span className="capitalize">{patientCase.status}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1 text-sm">
                      <p className="text-gray-700">
                        <span className="font-medium">Condition:</span> {patientCase.condition}
                      </p>
                      <p className="text-gray-700">
                        <span className="font-medium">ICD:</span> {patientCase.icdCode}
                      </p>
                      <p className="text-gray-500 text-xs mt-2">
                        Updated: {format(new Date(patientCase.updatedAt), 'MMM dd, yyyy')}
                      </p>
                    </div>
                    
                    {/* Actions */}
                    <div className="flex gap-2 mt-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onLoadCase(patientCase.id);
                          onClose();
                        }}
                        className="flex-1 px-3 py-2 bg-primary-50 text-primary-700 rounded text-sm font-medium hover:bg-primary-100 transition-colors"
                      >
                        Open
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm('Delete this case?')) {
                            onDeleteCase(patientCase.id);
                          }
                        }}
                        className="px-3 py-2 bg-red-50 text-red-700 rounded text-sm font-medium hover:bg-red-100 transition-colors"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;

