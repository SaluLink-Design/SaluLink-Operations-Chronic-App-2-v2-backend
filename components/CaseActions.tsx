'use client';

import { FileText, Repeat, UserPlus } from 'lucide-react';

interface CaseActionsProps {
  onOngoingManagement: () => void;
  onMedicationReport: () => void;
  onReferral: () => void;
}

const CaseActions = ({ onOngoingManagement, onMedicationReport, onReferral }: CaseActionsProps) => {
  return (
    <div className="card">
      <h2 className="text-xl font-bold text-gray-900 mb-6">Case Actions</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={onOngoingManagement}
          className="p-6 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all group text-left"
        >
          <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-primary-200 transition-colors">
            <Repeat className="w-6 h-6 text-primary-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Ongoing Management</h3>
          <p className="text-sm text-gray-600">
            Add ongoing treatment protocols and monitoring
          </p>
        </button>
        
        <button
          onClick={onMedicationReport}
          className="p-6 border-2 border-gray-200 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all group text-left"
        >
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-purple-200 transition-colors">
            <FileText className="w-6 h-6 text-purple-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Medication Report</h3>
          <p className="text-sm text-gray-600">
            Update medication status or prescribe new medication
          </p>
        </button>
        
        <button
          onClick={onReferral}
          className="p-6 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all group text-left"
        >
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
            <UserPlus className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Create Referral</h3>
          <p className="text-sm text-gray-600">
            Refer patient to specialist with case summary
          </p>
        </button>
      </div>
    </div>
  );
};

export default CaseActions;

