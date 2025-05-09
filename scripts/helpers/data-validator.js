const DataValidator = {
    validateData: (data) => {
        return data && 
               data.next_day && 
               data.Ledger_Information &&
               data.Ledger_Information.metrics;
    }
};
