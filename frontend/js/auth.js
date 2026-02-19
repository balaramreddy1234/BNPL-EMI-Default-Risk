function login() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const msgEl = document.getElementById("msg");

  if (!email || !password) {
    msgEl.innerText = "Email and password are required";
    return;
  }

  fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      msgEl.innerText = "";
      window.location.href = "/pages/dashboard.html";
    } else {
      msgEl.innerText = data.message || "Invalid email or password";
    }
  })
  .catch(err => {
    msgEl.innerText = "Error: " + err.message;
  });
}

function register() {
  const name = document.getElementById("name").value.trim();
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const confirmPassword = document.getElementById("confirm").value;
  const msgEl = document.getElementById("msg");

  msgEl.innerText = "";

  if (!name || !email || !password || !confirmPassword) {
    msgEl.innerText = "All fields are required";
    return;
  }

  if (password !== confirmPassword) {
    msgEl.innerText = "Passwords do not match";
    return;
  }

  if (password.length < 8) {
    msgEl.innerText = "Password must be at least 8 characters long";
    return;
  }

  const letterCount = (password.match(/[a-zA-Z]/g) || []).length;
  if (letterCount < 4) {
    msgEl.innerText = "Password must contain at least 4 letters";
    return;
  }

  const numberCount = (password.match(/[0-9]/g) || []).length;
  if (numberCount < 2) {
    msgEl.innerText = "Password must contain at least 2 numbers";
    return;
  }

  const specialCount = (password.match(/[^a-zA-Z0-9]/g) || []).length;
  if (specialCount < 2) {
    msgEl.innerText = "Password must contain at least 2 special characters";
    return;
  }

  fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      email,
      password,
      confirm_password: confirmPassword
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      msgEl.style.color = "green";
      msgEl.innerText = "Registration successful! Redirecting to login...";
      setTimeout(() => {
        window.location.href = "/pages/login.html";
      }, 2000);
    } else {
      msgEl.style.color = "red";
      msgEl.innerText = data.message || "Registration failed";
    }
  })
  .catch(err => {
    msgEl.style.color = "red";
    msgEl.innerText = "Error: " + err.message;
  });
}

function logout() {
  fetch("/logout", {
    method: "POST"
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      window.location.href = "/pages/login.html";
    }
  })
  .catch(err => {
    console.error("Logout error:", err);
    window.location.href = "/pages/login.html";
  });
}
