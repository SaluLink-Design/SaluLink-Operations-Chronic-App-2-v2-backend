// Core Types for SaluLink Chronic Treatment App

export interface ChronicCondition {
  condition: string;
  icdCode: string;
  icdDescription: string;
}

export interface MedicineItem {
  condition: string;
  cdaCore: string;
  cdaExecutive: string;
  medicineClass: string;
  activeIngredient: string;
  medicineNameAndStrength: string;
}

export interface TreatmentBasketItem {
  condition: string;
  diagnosticBasket: {
    description: string;
    code: string;
    covered: string;
  };
  ongoingManagementBasket: {
    description: string;
    code: string;
    covered: string;
  };
  specialists?: string;
}

export interface MatchedCondition {
  condition: string;
  icdCode: string;
  icdDescription: string;
  similarityScore: number;
}

export interface TreatmentItem {
  description: string;
  code: string;
  maxCovered: number;
  timesCompleted: number;
  documentation: {
    notes: string;
    images: string[];
  };
}

export interface SelectedMedication {
  medicineClass: string;
  activeIngredient: string;
  medicineNameAndStrength: string;
  cdaAmount: string;
}

export type MedicalPlan = 'Core' | 'Priority' | 'Saver' | 'Executive' | 'Comprehensive';

export interface PatientCase {
  id: string;
  patientName: string;
  patientId: string;
  createdAt: Date;
  updatedAt: Date;
  clinicalNote: string;
  condition: string;
  icdCode: string;
  icdDescription: string;
  diagnosticTreatments: TreatmentItem[];
  ongoingTreatments: TreatmentItem[];
  medications: SelectedMedication[];
  medicationNote: string;
  plan: MedicalPlan;
  status: 'draft' | 'diagnostic' | 'ongoing' | 'completed';
}

export interface ReferralData {
  caseId: string;
  urgency: 'low' | 'medium' | 'high';
  referralNote: string;
  specialistType: string;
}

export interface MedicationReport {
  caseId: string;
  originalMedications: SelectedMedication[];
  followUpNotes: string;
  newMedications?: SelectedMedication[];
  motivationLetter?: string;
}

export interface AnalysisResult {
  extractedKeywords: string[];
  matchedConditions: MatchedCondition[];
}

export interface WorkflowStep {
  id: string;
  title: string;
  completed: boolean;
  active: boolean;
}

