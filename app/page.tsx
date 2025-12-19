'use client';

import { useEffect, useState } from 'react';
import { useStore } from '@/lib/store';
import { DataService } from '@/lib/dataService';
import { PDFExportService } from '@/lib/pdfExport';
import { Menu, FileDown, Save, CheckCircle, ArrowLeft, ArrowRight } from 'lucide-react';

// Components
import ClinicalNoteInput from '@/components/ClinicalNoteInput';
import ConditionSelection from '@/components/ConditionSelection';
import IcdCodeSelection from '@/components/IcdCodeSelection';
import DiagnosticBasket from '@/components/DiagnosticBasket';
import MedicationSelection from '@/components/MedicationSelection';
import Sidebar from '@/components/Sidebar';
import CaseActions from '@/components/CaseActions';
import OngoingManagement from '@/components/OngoingManagement';
import MedicationReport from '@/components/MedicationReport';
import Referral from '@/components/Referral';
import { MatchedCondition } from '@/types';

type WorkflowMode = 'new' | 'ongoing' | 'medication' | 'referral';

export default function Home() {
  const store = useStore();
  const [isInitialized, setIsInitialized] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [matchedConditions, setMatchedConditions] = useState<MatchedCondition[]>([]);
  const [currentWorkflow, setCurrentWorkflow] = useState<WorkflowMode>('new');
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [patientId, setPatientId] = useState('');

  useEffect(() => {
    const init = async () => {
      await DataService.initialize();
      setIsInitialized(true);
    };
    init();
  }, []);

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clinical_note: store.clinicalNote }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      store.setExtractedKeywords(data.extracted_keywords || []);
      setMatchedConditions(data.matched_conditions || []);
      
      if (data.matched_conditions && data.matched_conditions.length > 0) {
        store.setCurrentStep(1);
      }
    } catch (error) {
      alert('Failed to analyze note. Please ensure the Python backend is running on port 8000.');
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectCondition = (condition: string, icdCode: string, description: string) => {
    store.setSelectedCondition(condition, icdCode, description);
  };

  const handleNextStep = () => {
    const nextStep = store.currentStep + 1;
    
    // Validation
    if (store.currentStep === 1 && !store.selectedCondition) {
      alert('Please select a condition');
      return;
    }
    if (store.currentStep === 2 && !store.selectedIcdCode) {
      alert('Please select an ICD-10 code');
      return;
    }
    
    store.setCurrentStep(nextStep);
  };

  const handlePreviousStep = () => {
    store.setCurrentStep(Math.max(0, store.currentStep - 1));
  };

  const handleExportPDF = () => {
    if (!store.selectedCondition || !store.selectedIcdCode) {
      alert('Please complete the workflow first');
      return;
    }

    const pdfService = new PDFExportService();
    const patientCase = {
      id: Date.now().toString(),
      patientName: patientName || 'Patient',
      patientId: patientId || 'N/A',
      createdAt: new Date(),
      updatedAt: new Date(),
      clinicalNote: store.clinicalNote,
      condition: store.selectedCondition,
      icdCode: store.selectedIcdCode,
      icdDescription: store.selectedIcdDescription || '',
      diagnosticTreatments: store.diagnosticTreatments,
      ongoingTreatments: store.ongoingTreatments,
      medications: store.medications,
      medicationNote: store.medicationNote,
      plan: store.selectedPlan,
      status: 'diagnostic' as const,
    };

    pdfService.exportInitialClaim(patientCase);
  };

  const handleSaveCase = () => {
    if (!patientName || !patientId) {
      alert('Please enter patient name and ID');
      return;
    }
    store.saveCase(patientName, patientId);
    setShowSaveModal(false);
    alert('Case saved successfully!');
  };

  const handleLoadCase = (caseId: string) => {
    store.loadCase(caseId);
    setCurrentWorkflow('new');
    const loadedCase = store.cases.find(c => c.id === caseId);
    if (loadedCase) {
      setPatientName(loadedCase.patientName);
      setPatientId(loadedCase.patientId);
      store.setCurrentStep(4); // Jump to view mode
    }
  };

  const handleOngoingManagementSave = () => {
    if (store.currentCaseId) {
      store.updateCase(store.currentCaseId, {
        ongoingTreatments: store.ongoingTreatments,
        status: 'ongoing',
      });
      
      const currentCase = store.cases.find(c => c.id === store.currentCaseId);
      if (currentCase) {
        const pdfService = new PDFExportService();
        pdfService.exportOngoingManagement(currentCase);
      }
    }
    alert('Ongoing management saved!');
    setCurrentWorkflow('new');
  };

  const handleMedicationReportSave = (followUpNotes: string, newMeds?: any[], motivationLetter?: string) => {
    if (store.currentCaseId) {
      const currentCase = store.cases.find(c => c.id === store.currentCaseId);
      if (currentCase) {
        const pdfService = new PDFExportService();
        pdfService.exportMedicationReport(currentCase, followUpNotes, newMeds, motivationLetter);
      }
    }
    alert('Medication report exported!');
    setCurrentWorkflow('new');
  };

  const handleReferralSave = (urgency: any, referralNote: string, specialistType: string) => {
    if (store.currentCaseId) {
      const currentCase = store.cases.find(c => c.id === store.currentCaseId);
      if (currentCase) {
        const pdfService = new PDFExportService();
        pdfService.exportReferral(currentCase, urgency, referralNote, specialistType);
      }
    }
    alert('Referral exported!');
    setCurrentWorkflow('new');
  };

  const steps = [
    { id: 0, title: 'Clinical Note' },
    { id: 1, title: 'Condition' },
    { id: 2, title: 'ICD Code' },
    { id: 3, title: 'Diagnostics' },
    { id: 4, title: 'Medication' },
  ];

  if (!isInitialized) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading SaluLink Chronic App...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">S</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">SaluLink</h1>
                <p className="text-xs text-gray-500">Chronic Treatment App</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {store.currentStep >= 3 && currentWorkflow === 'new' && (
                <>
                  <button
                    onClick={() => setShowSaveModal(true)}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Save className="w-5 h-5" />
                    Save Case
                  </button>
                  <button
                    onClick={handleExportPDF}
                    className="btn-primary flex items-center gap-2"
                  >
                    <FileDown className="w-5 h-5" />
                    Export PDF
                  </button>
                </>
              )}
              <button
                onClick={store.toggleSidebar}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <Menu className="w-6 h-6" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <Sidebar
        isOpen={store.sidebarOpen}
        onClose={store.toggleSidebar}
        cases={store.cases}
        onLoadCase={handleLoadCase}
        onDeleteCase={store.deleteCase}
      />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentWorkflow === 'new' && (
          <>
            {/* Progress Steps */}
            <div className="mb-8 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between">
                {steps.map((step, index) => (
                  <div key={step.id} className="flex items-center flex-1">
                    <div className="flex flex-col items-center flex-1">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                          store.currentStep > step.id
                            ? 'bg-green-500 text-white'
                            : store.currentStep === step.id
                            ? 'bg-primary-600 text-white'
                            : 'bg-gray-200 text-gray-600'
                        }`}
                      >
                        {store.currentStep > step.id ? (
                          <CheckCircle className="w-6 h-6" />
                        ) : (
                          step.id + 1
                        )}
                      </div>
                      <span className={`mt-2 text-sm font-medium ${
                        store.currentStep >= step.id ? 'text-gray-900' : 'text-gray-500'
                      }`}>
                        {step.title}
                      </span>
                    </div>
                    {index < steps.length - 1 && (
                      <div
                        className={`h-1 flex-1 mx-4 ${
                          store.currentStep > step.id ? 'bg-green-500' : 'bg-gray-200'
                        }`}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Step Content */}
            <div className="space-y-6">
              {store.currentStep === 0 && (
                <ClinicalNoteInput
                  value={store.clinicalNote}
                  onChange={store.setClinicalNote}
                  onAnalyze={handleAnalyze}
                  isAnalyzing={isAnalyzing}
                />
              )}

              {store.currentStep === 1 && (
                <ConditionSelection
                  matchedConditions={matchedConditions}
                  onSelect={handleSelectCondition}
                  selectedCondition={store.selectedCondition}
                />
              )}

              {store.currentStep === 2 && store.selectedCondition && (
                <IcdCodeSelection
                  condition={store.selectedCondition}
                  selectedIcdCode={store.selectedIcdCode}
                  onSelect={(code, desc) => {
                    store.setSelectedCondition(store.selectedCondition!, code, desc);
                  }}
                />
              )}

              {store.currentStep === 3 && store.selectedCondition && (
                <DiagnosticBasket
                  condition={store.selectedCondition}
                  treatments={store.diagnosticTreatments}
                  onAddTreatment={store.addDiagnosticTreatment}
                  onUpdateTreatment={store.updateDiagnosticTreatment}
                  onRemoveTreatment={(index) => {
                    const newTreatments = store.diagnosticTreatments.filter((_, i) => i !== index);
                    useStore.setState({ diagnosticTreatments: newTreatments });
                  }}
                />
              )}

              {store.currentStep === 4 && store.selectedCondition && (
                <>
                  <MedicationSelection
                    condition={store.selectedCondition}
                    selectedPlan={store.selectedPlan}
                    medications={store.medications}
                    medicationNote={store.medicationNote}
                    onAddMedication={store.addMedication}
                    onRemoveMedication={store.removeMedication}
                    onSetMedicationNote={store.setMedicationNote}
                    onSetPlan={store.setSelectedPlan}
                  />
                  
                  {store.currentCaseId && (
                    <CaseActions
                      onOngoingManagement={() => setCurrentWorkflow('ongoing')}
                      onMedicationReport={() => setCurrentWorkflow('medication')}
                      onReferral={() => setCurrentWorkflow('referral')}
                    />
                  )}
                </>
              )}

              {/* Navigation Buttons */}
              {store.currentStep > 0 && store.currentStep < 4 && (
                <div className="flex justify-between">
                  <button
                    onClick={handlePreviousStep}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <ArrowLeft className="w-5 h-5" />
                    Previous
                  </button>
                  <button
                    onClick={handleNextStep}
                    className="btn-primary flex items-center gap-2"
                  >
                    Next
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              )}

              {store.currentStep === 0 && matchedConditions.length > 0 && (
                <div className="flex justify-end">
                  <button
                    onClick={handleNextStep}
                    className="btn-primary flex items-center gap-2"
                  >
                    Next
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              )}
            </div>
          </>
        )}

        {currentWorkflow === 'ongoing' && store.selectedCondition && (
          <>
            <button
              onClick={() => setCurrentWorkflow('new')}
              className="btn-secondary flex items-center gap-2 mb-6"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Case
            </button>
            <OngoingManagement
              condition={store.selectedCondition}
              treatments={store.ongoingTreatments}
              onAddTreatment={store.addOngoingTreatment}
              onUpdateTreatment={store.updateOngoingTreatment}
              onRemoveTreatment={(index) => {
                const newTreatments = store.ongoingTreatments.filter((_, i) => i !== index);
                useStore.setState({ ongoingTreatments: newTreatments });
              }}
              onSave={handleOngoingManagementSave}
            />
          </>
        )}

        {currentWorkflow === 'medication' && store.currentCaseId && (
          <>
            <button
              onClick={() => setCurrentWorkflow('new')}
              className="btn-secondary flex items-center gap-2 mb-6"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Case
            </button>
            <MedicationReport
              currentMedications={store.medications}
              medicationNote={store.medicationNote}
              condition={store.selectedCondition!}
              selectedPlan={store.selectedPlan}
              onSave={handleMedicationReportSave}
            />
          </>
        )}

        {currentWorkflow === 'referral' && store.currentCaseId && (
          <>
            <button
              onClick={() => setCurrentWorkflow('new')}
              className="btn-secondary flex items-center gap-2 mb-6"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Case
            </button>
            {(() => {
              const currentCase = store.cases.find(c => c.id === store.currentCaseId);
              return currentCase ? (
                <Referral patientCase={currentCase} onSave={handleReferralSave} />
              ) : null;
            })()}
          </>
        )}
      </main>

      {/* Save Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold mb-4">Save Patient Case</h3>
            <div className="space-y-4">
              <div>
                <label className="label">Patient Name</label>
                <input
                  type="text"
                  className="input-field"
                  placeholder="Enter patient name"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                />
              </div>
              <div>
                <label className="label">Patient ID</label>
                <input
                  type="text"
                  className="input-field"
                  placeholder="Enter patient ID"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowSaveModal(false)}
                className="btn-secondary flex-1"
              >
                Cancel
              </button>
              <button onClick={handleSaveCase} className="btn-primary flex-1">
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

