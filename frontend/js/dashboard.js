/**
 * Dashboard Management
 * Features: Profile Modal, Support Modal, Photo Upload, and UI Sync
 */

document.addEventListener("DOMContentLoaded", () => {
    fetchUserProfile();
});

// 1. Fetch and Display User Profile (Syncs Dashboard, Modals, and Help form)
function fetchUserProfile() {
    fetch("/user/profile")
        .then(res => {
            if (res.status === 401) {
                window.location.href = "/pages/login.html";
                return;
            }
            return res.json();
        })
        .then(data => {
            if (data && data.name) {
                // Update Dashboard text
                document.getElementById("welcomeName").innerText = data.name;
                document.getElementById("dropdownFullName").innerText = data.name;
                document.getElementById("dropdownEmail").innerText = data.email;
                
                // Pre-fill Edit Modal fields
                document.getElementById("editName").value = data.name;
                document.getElementById("editAge").value = data.age || "";
                document.getElementById("editAddress").value = data.address || "";

                // Auto-fill Help & Support fields (ReadOnly in HTML)
                const helpName = document.getElementById("helpName");
                const helpEmail = document.getElementById("helpEmail");
                if (helpName) helpName.value = data.name;
                if (helpEmail) helpEmail.value = data.email;

                // Update Profile Icons (Nav bar & Modal Preview)
                const photoUrl = data.photo ? `/assets/profile_pics/${data.photo}` : null;
                const headerIcon = document.getElementById("headerProfileIcon");
                const modalPreview = document.getElementById("photoPreview");

                if (photoUrl) {
                    const imgHtml = `<img src="${photoUrl}" alt="Profile" style="width:100%; height:100%; object-fit:cover;">`;
                    headerIcon.innerHTML = imgHtml;
                    modalPreview.innerHTML = imgHtml;
                } else {
                    headerIcon.innerText = "👤";
                    modalPreview.innerText = "👤";
                }
            }
        })
        .catch(err => console.error("Could not load user profile:", err));
}

// 2. Help & Support Modal Controls
function openSupportModal() {
    const supportModal = document.getElementById("supportModal");
    if (supportModal) {
        supportModal.style.display = "block";
        toggleProfileMenu(); 
    }
}

function closeSupportModal() {
    const supportModal = document.getElementById("supportModal");
    if (supportModal) supportModal.style.display = "none";
}

// 3. Handle Help & Support Submission
async function submitHelpRequest(event) {
    event.preventDefault();
    
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerText;
    
    const payload = {
        name: document.getElementById("helpName").value,
        email: document.getElementById("helpEmail").value,
        category: document.getElementById("helpCategory").value,
        details: document.getElementById("helpDetails").value
    };

    try {
        submitBtn.innerText = "Sending...";
        submitBtn.disabled = true;

        const res = await fetch("/user/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const result = await res.json();
        if (result.success) {
            alert("✅ Support request submitted successfully!");
            document.getElementById("helpDetails").value = ""; 
            closeSupportModal();
        } else {
            alert("❌ Error: " + (result.error || "Submission failed."));
        }
    } catch (err) {
        alert("Server error. Please try again later.");
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
}

// 4. Profile Dropdown Toggle
function toggleProfileMenu() {
    const menu = document.getElementById("profileMenu");
    if (menu) {
        menu.style.display = (menu.style.display === "block") ? "none" : "block";
    }
}

// 5. Image Preview Logic
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('photoPreview');
            if (preview) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview" style="width:100%; height:100%; object-fit:cover;">`;
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// 6. Profile Modal Controls
function openEditModal() {
    const editModal = document.getElementById("editProfileModal");
    if (editModal) {
        editModal.style.display = "block";
        toggleProfileMenu(); 
    }
}

function closeModal() {
    const editModal = document.getElementById("editProfileModal");
    if (editModal) editModal.style.display = "none";
}

// 7. Save Profile Changes (Multipart/FormData for Photo)
async function saveProfileChanges() {
    const name = document.getElementById("editName").value;
    const age = document.getElementById("editAge").value;
    const address = document.getElementById("editAddress").value;
    const photoFile = document.getElementById("editPhoto").files[0];

    if (!name) {
        alert("Name cannot be empty");
        return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("age", age);
    formData.append("address", address);
    if (photoFile) formData.append("photo", photoFile);

    try {
        const res = await fetch("/user/update", {
            method: "POST",
            body: formData
        });

        const result = await res.json();
        if (result.success) {
            alert("Profile updated successfully!");
            closeModal();
            fetchUserProfile(); 
        } else {
            alert("Error: " + result.error);
        }
    } catch (err) {
        alert("Server error. Please try again.");
    }
}

// 8. Password Update
async function showChangePassword() {
    const newPassword = prompt("Enter your new password (min 6 chars):");
    if (!newPassword || newPassword.length < 6) {
        if (newPassword) alert("Password too short!");
        return;
    }

    const res = await fetch("/user/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: newPassword })
    });

    const result = await res.json();
    if (result.success) alert("Password changed successfully!");
}

// 9. Logout Function
function logout() {
    fetch("/logout")
        .then(() => window.location.href = "/pages/login.html");
}

// 10. Global window click handler
window.onclick = function(event) {
    if (!event.target.closest('.profile-dropdown')) {
        const menu = document.getElementById("profileMenu");
        if (menu && menu.style.display === "block") menu.style.display = "none";
    }
    
    const editModal = document.getElementById("editProfileModal");
    const supportModal = document.getElementById("supportModal");
    
    if (event.target === editModal) closeModal();
    if (event.target === supportModal) closeSupportModal();
};