import Papa from 'papaparse'; // TODO: Ensure 'papaparse' is installed with `npm install papaparse` and typing with `npm install --save-dev @types/papaparse`
import type { ChronicCondition, MedicineItem, TreatmentBasketItem } from '@/types';

// Parse CSV data from public folder
export class DataService {
  private static chronicConditions: ChronicCondition[] = [];
  private static medicines: MedicineItem[] = [];
  private static treatmentBasket: TreatmentBasketItem[] = [];
  private static initialized = false;

  static async initialize() {
    if (this.initialized) return;

    try {
      // Load Chronic Conditions
      const chronicResponse = await fetch('/Chronic Conditions.csv');
      const chronicText = await chronicResponse.text();
      const chronicParsed = Papa.parse<any>(chronicText, { header: true });
      
      this.chronicConditions = chronicParsed.data
        .filter(row => row['CHRONIC CONDITIONS'] && row['ICD-Code'])
        .map(row => ({
          condition: row['CHRONIC CONDITIONS'],
          icdCode: row['ICD-Code'],
          icdDescription: row['ICD-Code Description'] || '',
        }));

      // Load Medicine List
      const medicineResponse = await fetch('/Medicine List.csv');
      const medicineText = await medicineResponse.text();
      const medicineParsed = Papa.parse<any>(medicineText, { header: true });
      
      this.medicines = medicineParsed.data
        .filter(row => row['CHRONIC DISEASE LIST CONDITION'])
        .map(row => ({
          condition: row['CHRONIC DISEASE LIST CONDITION'],
          cdaCore: row['CDA FOR CORE, PRIORITY AND SAVER PLANS'] || '',
          cdaExecutive: row['CDA FOR EXECUTIVE AND COMPREHENSIVE PLANS'] || '',
          medicineClass: row['MEDICINE CLASS'] || '',
          activeIngredient: row['ACTIVE INGREDIENT'] || '',
          medicineNameAndStrength: row['MEDICINE NAME AND STRENGTH'] || '',
        }));

      // Load Treatment Basket
      const basketResponse = await fetch('/Treatment Basket.csv');
      const basketText = await basketResponse.text();
      const basketParsed = Papa.parse<any>(basketText, { header: true, skipEmptyLines: true });
      
      // Process treatment basket with proper column mapping
      let currentCondition = '';
      this.treatmentBasket = basketParsed.data
        .slice(1) // Skip header row
        .map((row: any) => {
          // Forward fill condition
          if (row['CONDITION']) {
            currentCondition = row['CONDITION'];
          }
          
          return {
            condition: currentCondition,
            diagnosticBasket: {
              description: row['DIAGNOSTIC BASKET'] || '',
              code: row['DIAGNOSTIC BASKET.1'] || '',
              covered: row['DIAGNOSTIC BASKET.2'] || '',
            },
            ongoingManagementBasket: {
              description: row['ONGOING MANAGEMENT BASKET'] || '',
              code: row['ONGOING MANAGEMENT BASKET.1'] || '',
              covered: row['ONGOING MANAGEMENT BASKET.2'] || '',
            },
            specialists: row['Unnamed: 7'] || '',
          };
        })
        .filter(item => item.condition);

      this.initialized = true;
    } catch (error) {
      console.error('Error initializing data service:', error);
      throw error;
    }
  }

  static getChronicConditions(): ChronicCondition[] {
    return this.chronicConditions;
  }

  static getConditionsByName(name: string): ChronicCondition[] {
    return this.chronicConditions.filter(c => 
      c.condition.toLowerCase().includes(name.toLowerCase())
    );
  }

  static getIcdCodesForCondition(condition: string): ChronicCondition[] {
    return this.chronicConditions.filter(c => 
      c.condition.toLowerCase() === condition.toLowerCase()
    );
  }

  static getMedicinesForCondition(condition: string): MedicineItem[] {
    return this.medicines.filter(m => 
      m.condition.toLowerCase() === condition.toLowerCase()
    );
  }

  static getTreatmentBasketForCondition(condition: string): TreatmentBasketItem[] {
    return this.treatmentBasket.filter(t => 
      t.condition.toLowerCase() === condition.toLowerCase()
    );
  }

  static getUniqueMedicineClasses(condition: string): string[] {
    const medicines = this.getMedicinesForCondition(condition);
    const classes = medicines.map(m => m.medicineClass).filter(Boolean);
    return Array.from(new Set(classes));
  }
}

