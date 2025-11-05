/*
 *	Copyright 2025 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

//go:build !js && !wasm

// Simple HTTP server for serving the WASM example.
package main

import (
	"flag"
	"log"
	"mime"
	"net/http"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP server address")
	flag.Parse()

	// Ensure .wasm files are served with the correct MIME type.
	_ = mime.AddExtensionType(".wasm", "application/wasm")

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(".")))

	log.Printf("Starting server at http://localhost%s", *addr)
	log.Println("Press Ctrl+C to stop")

	if err := http.ListenAndServe(*addr, logRequests(mux)); err != nil {
		log.Fatal(err)
	}
}

// logRequests wraps an HTTP handler to log incoming requests.
func logRequests(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		h.ServeHTTP(w, r)
	})
}
