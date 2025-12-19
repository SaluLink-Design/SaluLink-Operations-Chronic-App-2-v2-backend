'use client';

import { useState } from 'react';
import { UserPlus, AlertCircle } from 'lucide-react';
import { PatientCase } from '@/types';

interface ReferralProps {
  patientCase: PatientCase;
  onSave: (urgency: 'low' | 'medium' | 'high', referralNote: string, specialistType: string) => void;
}

const Referral = ({ patientCase, onSave }: ReferralProps) => {
  const [urgency, setUrgency] = useState<'low' | 'medium' | 'high'>('medium');
  const [referralNote, setReferralNote] = useState('');
  const [specialistType, setSpecialistType] = useState('');
  
  const urgencyOptions = [
    { value: 'low', label: 'Low', color: 'bg-green-100 text-green-700 border-green-300' },
    { value: 'medium', label: 'Medium', color: 'bg-yellow-100 text-yellow-700 border-yellow-300' },
    { value: 'high', label: 'High', color: 'bg-red-100 text-red-700 border-red-300' },
  ];
  
  return (
    <div className="card">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
          <UserPlus className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Create Referral</h2>
          <p className="text-sm text-gray-500">Refer patient to specialist</p>
        </div>
      </div>
      
      {/* Case Summary */}
      <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <h3 className="font-semibold text-lg mb-3">Case Summary</h3>
        <div className="space-y-2 text-sm">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-600">Patient:</p>
              <p className="font-medium">{patientCase.patientName}</p>
            </div>
            <div>
              <p className="text-gray-600">Patient ID:</p>
              <p className="font-medium">{patientCase.patientId}</p>
            </div>
          </div>
          <div>
            <p className="text-gray-600">Condition:</p>
            <p className="font-medium">{patientCase.condition}</p>
          </div>
          <div>
            <p className="text-gray-600">ICD-10 Code:</p>
            <p className="font-medium">{patientCase.icdCode} - {patientCase.icdDescription}</p>
          </div>
          <div>
            <p className="text-gray-600">Diagnostic Tests Completed:</p>
            <p className="font-medium">{patientCase.diagnosticTreatments.length}</p>
          </div>
          <div>
            <p className="text-gray-600">Current Medications:</p>
            <p className="font-medium">{patientCase.medications.length}</p>
          </div>
        </div>
      </div>
      
      {/* Specialist Type */}
      <div className="mb-4">
        <label className="label">Specialist Type</label>
        <input
          type="text"
          className="input-field"
          placeholder="e.g., Cardiologist, Pulmonologist, Nephrologist..."
          value={specialistType}
          onChange={(e) => setSpecialistType(e.target.value)}
        />
      </div>
      
      {/* Urgency */}
      <div className="mb-4">
        <label className="label">Urgency Level</label>
        <div className="flex gap-3">
          {urgencyOptions.map(option => (
            <button
              key={option.value}
              onClick={() => setUrgency(option.value as any)}
              className={`flex-1 px-4 py-3 rounded-lg border-2 font-medium transition-all ${
                urgency === option.value
                  ? option.color
                  : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Referral Note */}
      <div className="mb-6">
        <label className="label">Referral Motivation</label>
        <textarea
          className="textarea-field"
          rows={6}
          placeholder="Explain the reason for referral, clinical findings, and any specific concerns..."
          value={referralNote}
          onChange={(e) => setReferralNote(e.target.value)}
        />
      </div>
      
      {/* Warning */}
      {!referralNote.trim() && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-yellow-700">
            Please provide a detailed referral motivation to ensure proper specialist consultation.
          </p>
        </div>
      )}
      
      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={() => onSave(urgency, referralNote, specialistType)}
          disabled={!referralNote.trim() || !specialistType.trim()}
          className="btn-primary"
        >
          Create Referral
        </button>
      </div>
    </div>
  );
};

export default Referral;

