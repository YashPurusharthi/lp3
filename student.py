// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    // Structure to represent a student
    struct Student {
        string name;
        uint256 age;
    }

    // Array to store multiple students
    Student[] public students;

    // Event to log student data addition
    event StudentAdded(string name, uint256 age);

    // Counter for the number of times Ether is sent
    uint256 public etherSentCount;

    // Fallback function to receive Ether and increment the counter
    receive() external payable {
        etherSentCount++;
    }

    // Function to add a new student
    function addStudent(string memory name, uint256 age) public {
        Student memory newStudent = Student(name, age);
        students.push(newStudent);
        emit StudentAdded(name, age);
    }

    // Function to get the count of students
    function getStudentsCount() public view returns (uint256) {
        return students.length;
    }

    // Function to get a specific student's details by index
    function getStudent(uint256 index) public view returns (string memory, uint256) {
        require(index < students.length, "Student index out of range");
        Student memory student = students[index];
        return (student.name, student.age);
    }
}