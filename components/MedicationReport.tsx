'use client';

import { useState } from 'react';
import { SelectedMedication } from '@/types';
import { FileText, Plus } from 'lucide-react';
import MedicationSelection from './MedicationSelection';

interface MedicationReportProps {
  currentMedications: SelectedMedication[];
  medicationNote: string;
  condition: string;
  selectedPlan: any;
  onSave: (followUpNotes: string, newMedications?: SelectedMedication[], motivationLetter?: string) => void;
}

const MedicationReport = ({
  currentMedications,
  medicationNote,
  condition,
  selectedPlan,
  onSave
}: MedicationReportProps) => {
  const [followUpNotes, setFollowUpNotes] = useState('');
  const [addingNew, setAddingNew] = useState(false);
  const [newMedications, setNewMedications] = useState<SelectedMedication[]>([]);
  const [motivationLetter, setMotivationLetter] = useState('');
  
  const handleAddNewMedication = (medication: SelectedMedication) => {
    setNewMedications([...newMedications, medication]);
  };
  
  const handleRemoveNewMedication = (index: number) => {
    setNewMedications(newMedications.filter((_, i) => i !== index));
  };
  
  const handleSave = () => {
    onSave(
      followUpNotes,
      newMedications.length > 0 ? newMedications : undefined,
      newMedications.length > 0 ? motivationLetter : undefined
    );
  };
  
  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
            <FileText className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Medication Report</h2>
            <p className="text-sm text-gray-500">Review and update medication status</p>
          </div>
        </div>
        
        {/* Current Medications */}
        <div className="mb-6">
          <h3 className="font-semibold text-lg mb-3">Current Medications</h3>
          <div className="space-y-2">
            {currentMedications.map((med, index) => (
              <div key={index} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <p className="font-medium text-gray-900">{med.medicineNameAndStrength}</p>
                <p className="text-sm text-gray-600">{med.activeIngredient}</p>
                <p className="text-sm text-primary-600 font-medium">CDA: {med.cdaAmount}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm font-medium text-blue-900 mb-1">Registration Note:</p>
            <p className="text-sm text-blue-700">{medicationNote || 'No note provided'}</p>
          </div>
        </div>
        
        {/* Follow-up Notes */}
        <div className="mb-6">
          <label className="label">Follow-up Results & Effectiveness</label>
          <textarea
            className="textarea-field"
            rows={4}
            placeholder="Enter follow-up notes on medication effectiveness, patient response, side effects, etc..."
            value={followUpNotes}
            onChange={(e) => setFollowUpNotes(e.target.value)}
          />
        </div>
        
        {/* Add New Medication */}
        {!addingNew ? (
          <button
            onClick={() => setAddingNew(true)}
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Prescribe New Medication
          </button>
        ) : (
          <div className="space-y-4">
            <MedicationSelection
              condition={condition}
              selectedPlan={selectedPlan}
              medications={newMedications}
              medicationNote=""
              onAddMedication={handleAddNewMedication}
              onRemoveMedication={handleRemoveNewMedication}
              onSetMedicationNote={() => {}}
              onSetPlan={() => {}}
            />
            
            <div>
              <label className="label">Motivation Letter for Medication Change</label>
              <textarea
                className="textarea-field"
                rows={4}
                placeholder="Explain the reason for medication change or escalation..."
                value={motivationLetter}
                onChange={(e) => setMotivationLetter(e.target.value)}
              />
            </div>
            
            <button
              onClick={() => setAddingNew(false)}
              className="btn-secondary text-sm"
            >
              Cancel New Medication
            </button>
          </div>
        )}
        
        {/* Save Button */}
        <div className="mt-6 flex justify-end">
          <button onClick={handleSave} className="btn-primary">
            Save Medication Report
          </button>
        </div>
      </div>
    </div>
  );
};

export default MedicationReport;

