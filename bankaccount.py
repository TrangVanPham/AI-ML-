import random
from datetime import datetime

class BankAccount:
    def __init__(self, account_holder, account_type, initial_balance=0):
        """
        Initialize a new bank account
        
        Args:
            account_holder (str): Name of the account holder
            account_type (str): Type of account (e.g., 'Savings', 'Checking')
            initial_balance (float): Initial deposit amount (default 0)
        """
        self.account_holder = account_holder
        self.account_type = account_type
        self.balance = initial_balance
        self.account_number = self._generate_account_number()
        self.transaction_history = []
        
        # Record the initial deposit if any
        if initial_balance > 0:
            self._record_transaction("Initial Deposit", initial_balance)
    
    def _generate_account_number(self):
        """Generate a random 8-digit account number"""
        return random.randint(10000000, 99999999)
    
    def _record_transaction(self, transaction_type, amount):
        """Record a transaction in the history"""
        transaction = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": transaction_type,
            "amount": amount,
            "balance": self.balance
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount):
        """
        Deposit money into the account
        
        Args:
            amount (float): Amount to deposit
            
        Returns:
            str: Confirmation message
        """
        if amount <= 0:
            return "Deposit amount must be positive."
        
        self.balance += amount
        self._record_transaction("Deposit", amount)
        return f"Successfully deposited ${amount:.2f}. New balance: ${self.balance:.2f}"
    
    def withdraw(self, amount):
        """
        Withdraw money from the account
        
        Args:
            amount (float): Amount to withdraw
            
        Returns:
            str: Confirmation message or error message
        """
        if amount <= 0:
            return "Withdrawal amount must be positive."
        
        if amount > self.balance:
            return "Insufficient funds for this withdrawal."
        
        self.balance -= amount
        self._record_transaction("Withdrawal", -amount)
        return f"Successfully withdrew ${amount:.2f}. New balance: ${self.balance:.2f}"
    
    def check_balance(self):
        """Return the current account balance"""
        return f"Current balance: ${self.balance:.2f}"
    
    def get_account_type(self):
        """Return the account type"""
        return f"Account type: {self.account_type}"
    
    def get_account_number(self):
        """Return the account number"""
        return f"Account number: {self.account_number}"
    
    def get_holder_name(self):
        """Return the account holder's name"""
        return f"Account holder: {self.account_holder}"
    
    def get_transaction_history(self):
        """Return the transaction history"""
        if not self.transaction_history:
            return "No transactions yet."
        
        history = "Transaction History:\n"
        history += "-" * 50 + "\n"
        history += "Date                | Type           | Amount   | Balance\n"
        history += "-" * 50 + "\n"
        
        for transaction in self.transaction_history:
            history += (f"{transaction['date']} | {transaction['type']:<15} | "
                        f"${transaction['amount']:>7.2f} | ${transaction['balance']:>8.2f}\n")
        
        return history
    
    def __str__(self):
        """String representation of the account"""
        return (f"Bank Account\n{'-'*30}\n"
                f"Holder: {self.account_holder}\n"
                f"Account #: {self.account_number}\n"
                f"Type: {self.account_type}\n"
                f"Balance: ${self.balance:.2f}")


# Testing the BankAccount class
if __name__ == "__main__":
    # Create a new account
    account = BankAccount("John Doe", "Savings", 1000)
    print(account)
    print()
    
    # Test deposit
    print(account.deposit(500))
    print()
    
    # Test withdrawal
    print(account.withdraw(200))
    print()
    
    # Try to withdraw more than balance
    print(account.withdraw(2000))
    print()
    
    # Check balance
    print(account.check_balance())
    print()
    
    # Get account info
    print(account.get_account_number())
    print(account.get_account_type())
    print(account.get_holder_name())
    print()
    
    # Get transaction history
    print(account.get_transaction_history())